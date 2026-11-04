"""E25 klee100 load test: real-data path tests (R2-6 a/b). Quick (seconds); local and in the Kaggle units kit (Py3.12).

The unit tests' A/A and A-vs-C checks compare byte-identical files. These tests run the notebook's REAL parsing / loading /
comparison code (picked from k100load_code_export.py with the same AST mechanism as k100load_unit_tests.py) on NB0's real
arrays, rebuilt into vLLM 0.32 OpenAI-completions JSON and sent through the export's own _post (urllib stubbed, not _post).

  R1 TF parse          NB0's A / B teacher-forced npz -> a vLLM 0.32 /v1/completions response (prompt_logprobs: position 0
                       None, then {"<id>": {"logprob", "rank", "decoded_token": "token_id:<id>"}} with the forced token always
                       present, top-k ranks 1..k) -> _post -> tf_request -> pl_arrays: ids / lp / rank / top1 / top1_lp /
                       topk_ids / topk_lp / valid equal the npz exactly (also the npz tf_request writes to tf_raw/); the
                       request body; keys "token_id:<id>" (return_tokens_as_token_ids) parse the same
  R2 k = 1 retry       HTTP 400 "prompt-scoring tensor bound" -> a second request with prompt_logprobs 1, recorded; arrays =
                       the npz (top-k column 0); any other HTTP error -> no retry, error recorded, no arrays
  R3 malformed         no choices / choices empty / not JSON -> error; prompt_logprobs null / absent / a dict / cut short /
                       all None, the forced token absent, a non-numeric logprob -> never a MEASURED comparison, and an error
                       status on the request (tf_gate stops / tf_core_done False) instead of silent all-invalid arrays
  R4 decode parse      NB0's decode npz -> a completions response with logprobs (tokens / token_logprobs / top_logprobs) ->
                       decode(): tok / lp / top1 / top1_lp / top2_lp equal the npz; a short decode -> INCOMPLETE_DECODE
  R5 ref_load          the real dataset folder -> ok True, every pinned file, TFR / SCR = the npz arrays; every file the
                       reference json names is pinned; the pin = the dataset's SHA256SUMS.json; no dataset -> ok False
  R6 ref_load negative a temp copy: one npz truncated, one deleted, a pin (SHA256SUMS) mismatch, an ids-sha mismatch behind a
                       re-pinned json -> ok False with a why naming the file
  R7 AB reproduction   NB0's B arrays (FP8 KV, MTP off) as boot K -> k100_analysis: per band full + last256 mean / p99 /
                       rank-1 = NB0's recorded AB yardstick, yardstick ratio 1.0 / delta 0, labels, reading; K's A/A = NB0's
                       B A/A; K0 = B too -> K0 vs K = 0 and the headline from K0
  R8 SC reproduction   NB0's B decode + decode-TF npz through decode() / tf_request() (the parse path) as SCR['K'] ->
                       k100_analysis self_consistency k_decode_vs_tf = NB0's recorded B decode_vs_tf per band
Fixtures resolve from the repo root (this file's folder, 4 levels up) or env K100_TEST_*; a missing fixture is a SKIP with
its reason. A function missing from the export is a FAIL naming it.
Default: R1 parses 2k / 8k (A) and 2k (B), R8 parses the 2k / 8k decode-TF; K100_RD_FULL=1 adds the 40k / 90k responses (~30 s).
Run: python k100load_tests_realdata.py   (prints 'N/M PASS' and 'K skipped')
"""
import ast
import hashlib
import io
import json
import math
import os
import runpy
import shutil
import statistics
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import types
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k100load_lib as KL  # noqa: E402

REPO = HERE.parents[3] if len(HERE.parents) > 3 else HERE


def fixture(env, rel):
    return Path(os.environ.get(env) or (REPO / rel))


EXPORT = Path(os.environ.get("K100_TEST_EXPORT") or HERE / "k100load_code_export.py")
PAYLOAD = Path(os.environ.get("K100_TEST_PAYLOAD") or HERE / "k100load_payload")
REF = fixture("K100_TEST_REF", "_scratch/push_e25/ds_e25_nb0_tf_ref")
NB0_RAW = fixture("K100_TEST_NB0_RAW", "_scratch/e25_gpu_out/q27-xhigh-e25-nb0-fp8-quality/nb0/tf_raw")
BANDS = ("2k", "8k", "40k", "90k")
FULL = os.environ.get("K100_RD_FULL") == "1"        # 1: also parse the 40k / 90k responses (about 30 s more)
TMP = Path(tempfile.mkdtemp(prefix="k100rd_"))
RES, SKIPS = [], []


def check(name, cond, info=""):
    RES.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  | {str(info)[:600]}"), flush=True)


def skip(name, why):
    SKIPS.append((name, why))
    print(f"SKIP {name}: {why}", flush=True)


def have(path, name):
    if Path(path).exists():
        return True
    skip(name, f"fixture not present: {path}")
    return False


class Missing(Exception):
    pass


def run(name, fn):
    t0 = time.time()
    try:
        fn()
        print(f"  ({name}: {time.time() - t0:.1f} s)", flush=True)
    except Missing as e:
        RES.append((name, False))
        print(f"FAIL {name}: {e}", flush=True)
    except Exception as e:
        RES.append((name, False))
        print(f"FAIL {name} raised {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}", flush=True)


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- the export's own code (k100load_unit_tests.py's mechanism)
EXPORT_SRC = EXPORT.read_text(encoding="utf-8")
_TREE = ast.parse(EXPORT_SRC)


_DEFS = {}
for _node in _TREE.body:
    for _n in ([_node] + (_node.body if isinstance(_node, ast.If) else [])):
        if isinstance(_n, (ast.FunctionDef, ast.ClassDef)):
            _DEFS.setdefault(_n.name, []).append(_n)
        elif isinstance(_n, ast.Assign):
            for _x in {x.id for t in _n.targets for x in ast.walk(t) if isinstance(x, ast.Name)}:
                _DEFS.setdefault(_x, []).append(_n)
_CODE = {}


def defs(name):
    """Every (def / class / assignment) node defining `name` at module level or one level inside an `if` block."""
    return _DEFS.get(name, [])


def need(names):
    miss = [n for n in names if not defs(n)]
    if miss:
        raise Missing("function / name missing in the export: " + ", ".join(miss))


def load(names, ns):
    """exec the export's own (last) definition of each name into ns (compiled once)."""
    need(names)
    for n in names:
        if n not in _CODE:
            seg = ast.get_source_segment(EXPORT_SRC, defs(n)[-1], padded=True)
            _CODE[n] = compile(textwrap.dedent(seg), f"<export {n}>", "exec")
        exec(_CODE[n], ns)
    return ns


def export_const(name):
    nodes = [n for n in ast.walk(_TREE) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                                          for t in n.targets)]
    if not nodes:
        raise Missing(f"constant {name} missing in the export")
    return ast.literal_eval(nodes[-1].value)


_LAB = {}


def lab():
    """The lab the notebook loads (sha = the export's K100_LAB_SHA): the payload copy, else the reference dataset copy."""
    if "L" in _LAB:
        return _LAB["L"]
    want = export_const("K100_LAB_SHA")
    for p in (PAYLOAD / "k100load_lab.py", REF / "k100load_lab.py"):
        if p.is_file() and sha(p.read_bytes()) == want:
            q = TMP / "k100_lab_from_export.py"
            q.write_bytes(p.read_bytes())
            _LAB["L"] = types.SimpleNamespace(**runpy.run_path(str(q), run_name="k100_lab_test"))
            return _LAB["L"]
    _LAB["L"] = None
    return None


# ---------------------------------------------------------------- the fake HTTP layer (urlopen only: the export's _post runs)
class FakeHTTP:
    def __init__(self):
        self.queue, self.bodies, self.urls = [], [], []

    def urlopen(self, req, timeout=None):
        self.urls.append(req.full_url)
        self.bodies.append(json.loads(req.data))
        item = self.queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return io.BytesIO(item if isinstance(item, bytes) else json.dumps(item).encode())


def http_error(code, msg):
    return urllib.error.HTTPError("http://127.0.0.1:1/v1/completions", code, "err", {}, io.BytesIO(msg.encode()))


TF_NAMES = ["NB0", "SCREEN", "TRIAGE", "_finite", "_post", "_req_log", "ids_sha", "_cached", "pl_arrays", "e23_rows",
            "tf_request", "decode", "cmp_last256", "chunk_check", "cmp_full", "shift_check", "screen", "sc_compare",
            "decode_compare", "tf_gate"]


def tf_ns(fake):
    L = lab()
    if L is None:
        return None
    raw = TMP / f"raw_{len(os.listdir(TMP))}"
    raw.mkdir()
    ns = {"np": np, "math": math, "json": json, "hashlib": hashlib, "statistics": statistics, "time": time, "os": os,
          "L": L, "KL": KL, "UR": types.SimpleNamespace(Request=urllib.request.Request, urlopen=fake.urlopen), "UE": urllib.error,
          "T_START": time.time(), "RAW_DIR": raw, "NB0_REQ": raw / "k100_requests.jsonl", "_SAVE_LOCK": threading.Lock()}
    return load(TF_NAMES, ns)


def npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def kf_int(t):
    return str(int(t))                    # vLLM: dict[int, Logprob] serialised -> the id as a str


def kf_tok(t):
    return f"token_id:{int(t)}"           # the other key form the parser accepts


def tf_response(z, k=None, kf=kf_int, n_pl=None):
    """A vLLM 0.32 /v1/completions response (max_tokens 1, prompt_logprobs k, return_tokens_as_token_ids) whose
    prompt_logprobs carry exactly the npz's numbers: position 0 None; per valid position the forced token (its logprob /
    rank) and the top-k entries (ranks 1..k, overwriting the forced entry when the same id, as vLLM's dict build does)."""
    ids, n = z["ids"], len(z["ids"])
    k = z["topk_ids"].shape[1] if k is None else k
    pl = [None]
    for p in range(1, n):
        if not z["valid"][p]:
            pl.append(None)
            continue
        f = int(ids[p])
        d = {kf(f): {"logprob": float(z["lp"][p]), "rank": int(z["rank"][p]), "decoded_token": f"token_id:{f}"}}
        for j in range(k):
            t = int(z["topk_ids"][p, j])
            if t >= 0:
                d[kf(t)] = {"logprob": float(z["topk_lp"][p, j]), "rank": j + 1, "decoded_token": f"token_id:{t}"}
        pl.append(d)
    if n_pl is not None:
        pl = pl[:n_pl]
    return {"id": "cmpl-k100test", "object": "text_completion", "created": 0, "model": "q27",
            "choices": [{"index": 0, "text": "token_id:198", "token_ids": None, "logprobs": None, "finish_reason": "length",
                         "stop_reason": None, "prompt_logprobs": pl, "prompt_token_ids": None}],
            "usage": {"prompt_tokens": n, "total_tokens": n + 1, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}}}


KEYS = ("ids", "lp", "rank", "top1", "top1_lp", "topk_ids", "topk_lp", "valid")


def same_arrays(a, z, keys=KEYS, k=None):
    bad = []
    for x in keys:
        want = z[x][:, :k] if (k is not None and x in ("topk_ids", "topk_lp")) else z[x]
        got = a.get(x)
        if got is None or got.dtype != want.dtype or got.shape != want.shape or \
                not np.array_equal(got, want, equal_nan=got.dtype.kind == "f"):
            bad.append(x)
    return bad


def ref_meta():
    return json.loads((REF / "nb0_reference.json").read_text())


# ======================================================================================== R1 TF response parsing
def r1():
    if not have(REF / "A_2k_fresh.npz", "R1 TF parse"):
        return
    meta = ref_meta()
    files = [("A", b, REF / f"A_{b}_fresh.npz") for b in (BANDS if FULL else ("2k", "8k"))]
    if not FULL:
        skip("R1 TF parse at 40k / 90k", "set K100_RD_FULL=1 (about 30 s)")
    if (NB0_RAW / "B_2k_fresh.npz").is_file():
        files += [("B", "2k", NB0_RAW / "B_2k_fresh.npz")] + ([("B", "40k", NB0_RAW / "B_40k_fresh.npz")] if FULL else [])
    else:
        skip("R1 TF parse on B arrays", f"local NB0 output not present: {NB0_RAW}")
    for boot, band, f in files:
        z = npz(f)
        fake = FakeHTTP()
        ns = tf_ns(fake)
        if ns is None:
            skip("R1 TF parse", "no lab copy with the export's K100_LAB_SHA (payload / dataset)")
            return
        fake.queue.append(json.dumps(tf_response(z)).encode())
        t0 = time.time()
        r = ns["tf_request"]("K", 8001, [int(x) for x in z["ids"]], band, "fresh")
        m, a = r["meta"], r["arr"]
        bad = same_arrays(a, z) if a is not None else ["arr None"]
        check(f"R1 {boot}_{band}: vLLM 0.32 JSON -> _post -> tf_request -> pl_arrays = the npz (all 8 arrays, dtypes)", not bad,
              (bad, m.get("error")))
        n = len(z["ids"])
        rec_sha = meta["tf"]["A"][band]["prompt_sha256"]
        check(f"R1 {boot}_{band}: meta valid = expected = n-1, nonfinite 0, k 5, no note / error, prompt sha = NB0's",
              m.get("valid") == n - 1 == m.get("expected") and m.get("nonfinite") == 0 and m.get("k") == 5 and m.get("note") is None
              and m.get("error") is None and m.get("prompt_sha256") == rec_sha and m.get("prompt_logprobs_len") == n
              and m.get("top1_ambiguous") == 0 and m.get("cached_tokens") == 0, m)
        b = fake.bodies[0]
        check(f"R1 {boot}_{band}: request = /v1/completions, prompt ids, max_tokens 1, T 0, prompt_logprobs 5, token ids",
              len(fake.bodies) == 1 and fake.urls[0].endswith(":8001/v1/completions") and b["prompt"] == [int(x) for x in z["ids"]]
              and b["max_tokens"] == 1 and b["temperature"] == 0 and b["prompt_logprobs"] == 5 and b["return_tokens_as_token_ids"] is True,
              {k: v for k, v in b.items() if k != "prompt"})
        saved = ns["RAW_DIR"] / f"K_{band}_fresh.npz"
        check(f"R1 {boot}_{band}: the npz tf_request writes (tf_raw/{saved.name}) = the source npz",
              saved.is_file() and m.get("raw") == saved.name and not same_arrays(npz(saved), z), m.get("raw"))
        if band == "2k":
            lines = ns["NB0_REQ"].read_text().splitlines()
            check(f"R1 {boot}_{band}: the request is logged (k100_requests.jsonl, kind teacher_forced, no prompt copy)",
                  len(lines) == 1 and json.loads(lines[0])["kind"] == "teacher_forced" and "prompt" not in json.loads(lines[0])["body"])
            fake.queue.append(json.dumps(tf_response(z, kf=kf_tok)).encode())
            r2 = ns["tf_request"]("K", 8001, [int(x) for x in z["ids"]], band, "repeat")
            check(f"R1 {boot}_{band}: keys 'token_id:<id>' parse the same", r2["arr"] is not None and not same_arrays(r2["arr"], z),
                  r2["meta"].get("error"))
            full = ns["cmp_full"](z, a)
            check(f"R1 {boot}_{band}: parsed vs npz cmp_full = 0 / 1.0 MEASURED (n = expected)",
                  full["status"] == "MEASURED" and full["mean_abs_dlogprob"] == 0 and full["rank1_agreement"] == 1.0
                  and full["n"] == full["expected"] == n - 1, full)
        print(f"  ({boot}_{band}: {time.time() - t0:.1f} s)", flush=True)


# ======================================================================================== R2 k = 1 retry
def r2():
    if not have(REF / "A_8k_fresh.npz", "R2 k=1 retry"):
        return
    z = npz(REF / "A_8k_fresh.npz")
    ids = [int(x) for x in z["ids"]]
    fake = FakeHTTP()
    ns = tf_ns(fake)
    if ns is None:
        skip("R2 k=1 retry", "no lab copy with the export's K100_LAB_SHA")
        return
    fake.queue += [http_error(400, '{"error": {"message": "prompt_logprobs 5 exceeds the prompt-scoring tensor bound (8192 x 2)"}}'),
                   json.dumps(tf_response(z, k=1)).encode()]
    r = ns["tf_request"]("K", 8001, ids, "8k", "fresh")
    m, a = r["meta"], r["arr"]
    check("R2 k>1 refused (HTTP 400 'prompt-scoring tensor bound') -> retried once with prompt_logprobs 1",
          [b["prompt_logprobs"] for b in fake.bodies] == [5, 1] and m.get("k") == 1 and m.get("error") is None
          and "retried with 1" in (m.get("note") or "") and "HTTP 400" in (m.get("note") or ""), (m, [b["prompt_logprobs"] for b in fake.bodies]))
    bad = same_arrays(a, z, k=1) if a is not None else ["arr None"]
    check("R2 k=1 arrays: ids / lp / rank / top1 / top1_lp / valid = the npz, topk = its column 0 (shape n x 1)",
          not bad and a["topk_ids"].shape == (len(ids), 1), bad)
    fake2 = FakeHTTP()
    ns2 = tf_ns(fake2)
    fake2.queue += [http_error(500, "engine dead")]
    r = ns2["tf_request"]("K", 8001, ids, "8k", "fresh")
    check("R2 another HTTP error (500) -> no retry, 'HTTP 500: engine dead' recorded, no arrays, tf_gate stops",
          len(fake2.bodies) == 1 and r["arr"] is None and r["meta"]["error"] == "HTTP 500: engine dead"
          and (ns2["tf_gate"]({"8k": {"fresh": r["meta"]}}) or "").startswith("NO_GO_TF_REQUEST_ERROR"), r["meta"])
    fake3 = FakeHTTP()
    ns3 = tf_ns(fake3)
    fake3.queue += [http_error(400, "prompt-scoring tensor bound"), http_error(400, "prompt-scoring tensor bound")]
    r = ns3["tf_request"]("K", 8001, ids, "8k", "fresh", k=1)
    check("R2 k=1 refused -> no second retry, the error recorded", len(fake3.bodies) == 1 and r["arr"] is None
          and "HTTP 400" in (r["meta"]["error"] or ""), r["meta"])


# ======================================================================================== R3 malformed responses
def r3():
    if not have(REF / "A_2k_fresh.npz", "R3 malformed"):
        return
    z = npz(REF / "A_2k_fresh.npz")
    ids = [int(x) for x in z["ids"]]
    n = len(ids)
    good = tf_response(z)

    def go(item):
        fake = FakeHTTP()
        ns = tf_ns(fake)
        fake.queue.append(item)
        r = ns["tf_request"]("K", 8001, ids, "2k", "fresh")
        return ns, r

    if lab() is None:
        skip("R3 malformed", "no lab copy with the export's K100_LAB_SHA")
        return
    # hard errors: the request record carries an error, no arrays, tf_gate stops the boot
    for lab_, item in (("no 'choices' (an error object with HTTP 200)", {"object": "error", "message": "boom", "code": 500}),
                       ("not JSON (an HTML page)", b"<html>502 Bad Gateway</html>"),
                       ("choices empty", dict(good, choices=[]))):
        ns, r = go(item)
        g = ns["tf_gate"]({"2k": {"fresh": r["meta"]}})
        check(f"R3 {lab_} -> error recorded, no arrays, tf_gate NO_GO_TF_REQUEST_ERROR, tf_core_done False",
              r["arr"] is None and bool(r["meta"].get("error")) and (g or "").startswith("NO_GO_TF_REQUEST_ERROR")
              and not KL.tf_core_done({b: {"fresh": r["meta"]} for b in ("2k", "8k", "40k")}), (r["meta"].get("error"), g))
    # incomplete / non-numeric: never a MEASURED comparison (no silent zeros), and the request must not read as complete
    ch = good["choices"][0]
    pl = ch["prompt_logprobs"]
    pl_noforced = list(pl)
    pl_noforced[700] = {k: v for k, v in pl[700].items() if k != str(ids[700])} or {"0": {"logprob": -1.0, "rank": 1}}
    pl_nan = list(pl)
    pl_nan[900] = {k: dict(v, logprob="nan") if k == str(ids[900]) else v for k, v in pl[900].items()}
    cases = (("prompt_logprobs missing (null)", dict(ch, prompt_logprobs=None), "INCONCLUSIVE_NO_VALID"),
             ("prompt_logprobs a dict, not a list", dict(ch, prompt_logprobs={"1": pl[1]}), "INCONCLUSIVE_NO_VALID"),
             ("prompt_logprobs absent", {k: v for k, v in ch.items() if k != "prompt_logprobs"}, "INCONCLUSIVE_NO_VALID"),
             ("prompt_logprobs cut to half the prompt", dict(ch, prompt_logprobs=pl[:n // 2]), "PARTIAL"),
             ("every position None", dict(ch, prompt_logprobs=[None] * n), "INCONCLUSIVE_NO_VALID"),
             ("the forced token absent at one position", dict(ch, prompt_logprobs=pl_noforced), "PARTIAL"),
             ("a non-numeric logprob at one position", dict(ch, prompt_logprobs=pl_nan), "INCONCLUSIVE_NONFINITE"))
    for lab_, c, want in cases:
        ns, r = go(dict(good, choices=[c]))
        m, a = r["meta"], r["arr"]
        full = ns["cmp_full"](z, a) if a is not None else {"status": "INCONCLUSIVE_MISSING_TF"}
        lbl = KL.tf_label(full)
        check(f"R3 {lab_} -> no MEASURED comparison ({want}), label INCONCLUSIVE_*",
              full.get("status") == want and lbl.startswith("INCONCLUSIVE"), (full.get("status"), lbl, m.get("valid"), m.get("expected")))
        g = ns["tf_gate"]({"2k": {"fresh": m}})
        core = KL.tf_core_done({b: {"fresh": m} for b in ("2k", "8k", "40k")})
        check(f"R3 {lab_} -> an error status on the request (tf_gate stops / tf_core_done False), not silent "
              f"{'all-invalid' if want == 'INCONCLUSIVE_NO_VALID' else 'partial'} arrays",
              bool(m.get("error")) or bool(g) or not core,
              {"error": m.get("error"), "valid": m.get("valid"), "expected": m.get("expected"), "nonfinite": m.get("nonfinite"),
               "tf_gate": g, "tf_core_done": core})


# ======================================================================================== R4 decode response parsing
def dec_response(z, n_tok=None):
    """A vLLM 0.32 completions response for decode(): token_ids, logprobs.tokens ('token_id:<id>'), token_logprobs,
    top_logprobs ({'token_id:<id>': lp}: the rank-1 id, the chosen token, and a rank-2 entry at top2_lp)."""
    tok = [int(x) for x in z["tok"]][:n_tok]
    tops = []
    for i, t in enumerate(tok):
        d = {f"token_id:{int(z['top1'][i])}": float(z["top1_lp"][i])}
        if t != int(z["top1"][i]):
            d[f"token_id:{t}"] = float(z["lp"][i])
        if len(d) < 2 or not any(v == float(z["top2_lp"][i]) for k, v in d.items() if k != f"token_id:{int(z['top1'][i])}"):
            d[f"token_id:{int(z['top1'][i]) - 1 if int(z['top1'][i]) > 0 else 1}"] = float(z["top2_lp"][i])
        tops.append(d)
    return {"id": "cmpl-dec", "object": "text_completion", "created": 0, "model": "q27",
            "choices": [{"index": 0, "text": "x", "token_ids": tok, "finish_reason": "length", "stop_reason": None,
                         "logprobs": {"tokens": [f"token_id:{t}" for t in tok], "token_logprobs": [float(x) for x in z["lp"][:len(tok)]],
                                      "top_logprobs": tops, "text_offset": list(range(len(tok)))}}],
            "usage": {"prompt_tokens": len(z["ids"]), "completion_tokens": len(tok), "total_tokens": len(z["ids"]) + len(tok),
                      "prompt_tokens_details": {"cached_tokens": 0}}}


def r4():
    if not have(REF / "A_dec2k_cold.npz", "R4 decode parse"):
        return
    if lab() is None:
        skip("R4 decode parse", "no lab copy with the export's K100_LAB_SHA")
        return
    for f in ("A_dec2k_cold.npz", "A_dec40k_cold.npz"):
        z = npz(REF / f)
        fake = FakeHTTP()
        ns = tf_ns(fake)
        fake.queue.append(json.dumps(dec_response(z)).encode())
        out = ns["decode"]("K", 8001, [int(x) for x in z["ids"]], len(z["tok"]), "2k", "cold")
        bad = [k for k, x in (("tok", np.asarray(out["tok"], np.int32)), ("lp", out["lp"]), ("top1", out["top1"]),
                              ("top1_lp", out["top1_lp"]), ("top2_lp", out["top2_lp"]))
               if x.shape != z[k].shape or not np.array_equal(x.astype(z[k].dtype), z[k], equal_nan=z[k].dtype.kind == "f")]
        b = fake.bodies[0]
        check(f"R4 {f}: completions logprobs JSON -> decode(): tok / lp / top1 / top1_lp / top2_lp = the npz, no error",
              not bad and out["meta"]["error"] is None and out["meta"]["tokens"] == len(z["tok"]) and b["ignore_eos"] is True
              and b["temperature"] == 0 and b["logprobs"] == 5, (bad, out["meta"].get("error")))
    z = npz(REF / "A_dec2k_cold.npz")
    fake = FakeHTTP()
    ns = tf_ns(fake)
    fake.queue.append(json.dumps(dec_response(z, n_tok=100)).encode())
    out = ns["decode"]("K", 8001, [int(x) for x in z["ids"]], len(z["tok"]), "2k", "cold")
    check("R4 a decode cut at 100 of 256 tokens -> INCOMPLETE_DECODE, no npz written",
          (out["meta"]["error"] or "").startswith("INCOMPLETE_DECODE") and "raw" not in out["meta"], out["meta"].get("error"))


# ======================================================================================== R5 / R6 ref_load
def ref_ns(ref_dir, pin=None):
    L = lab()
    ns = {"np": np, "json": json, "hashlib": hashlib, "math": math, "statistics": statistics, "os": os, "L": L, "KL": KL,
          "REF_DIR": ref_dir, "TFR": {}, "SCR": {}, "REF": {}, "K1": {}, "RECS": {}}
    load(["K100_PIN"], ns)
    if pin is not None:
        ns["K100_PIN"] = pin
    load(["NB0", "SCREEN", "TRIAGE", "_finite", "ids_sha", "e23_rows", "cmp_last256", "chunk_check", "cmp_full", "shift_check",
          "screen", "sc_compare", "decode_compare", "RC2_WINDOWS", "ref_load", "_arr", "cmp_pair", "tf_skips", "k100_analysis"], ns)
    return ns


def r5():
    if not have(REF / "nb0_reference.json", "R5 ref_load"):
        return
    if lab() is None:
        skip("R5 ref_load", "no lab copy with the export's K100_LAB_SHA")
        return
    ns = ref_ns(REF)
    out = ns["ref_load"]()
    pin = ns["K100_PIN"]["ref_sha256"]
    check("R5 ref_load on the real dataset folder -> ok True, no bad, every pinned file", out.get("ok") is True and out.get("bad") == []
          and out.get("why") is None and out.get("files") == len(pin), {k: out.get(k) for k in ("ok", "why", "bad", "files")})
    meta = ref_meta()
    bad = []
    for key, bands in meta["tf"].items():
        for band, rec in bands.items():
            for run_ in ("fresh", "repeat"):
                got = ns["TFR"][key][band][run_]
                if same_arrays(got["arr"], npz(REF / rec[run_]["file"])) or got["meta"]["prompt_sha256"] != rec["prompt_sha256"]:
                    bad.append(f"{key} {band} {run_}")
    check("R5 TFR (A 2k-90k, C 2k-40k, fresh + repeat) = the npz arrays, prompt sha = NB0's", not bad and
          sorted(ns["TFR"]) == ["A", "C"] and sorted(ns["TFR"]["A"]) == sorted(BANDS), bad)
    sbad = [b for b, rec in meta["sc"]["A"].items() if ns["SCR"]["A"][b]["cold"]["tok"] != [int(x) for x in npz(REF / rec["cold"])["tok"]]
            or ns["SCR"]["A"][b]["P"] != rec["prompt_len"]]
    check("R5 SCR['A'] (2k / 8k / 40k) = the decode npz", not sbad and sorted(ns["SCR"]["A"]) == ["2k", "40k", "8k"], sbad)
    named = {r[x]["file"] for v in meta["tf"].values() for r in v.values() for x in ("fresh", "repeat")}
    named |= {r[x] for r in meta["sc"]["A"].values() for x in ("cold", "cached", "tf")}
    named.add("nb0_reference.json")
    check("R5 every file the reference json names is pinned (no unverified load)", named <= set(pin), sorted(named - set(pin)))
    sums = json.loads((REF / "SHA256SUMS.json").read_text())
    check("R5 the embedded pin = the dataset's SHA256SUMS.json for every pinned file", all(sums.get(n) == h for n, h in pin.items()),
          [n for n, h in pin.items() if sums.get(n) != h])
    real = {n: sha((REF / n).read_bytes()) for n in pin}
    check("R5 the pin = the files' real sha256", real == pin, [n for n in pin if real[n] != pin[n]])
    out = ref_ns(None)["ref_load"]()
    check("R5 no dataset attached (REF_DIR None) -> ok False with a why", out.get("ok") is False and "not attached" in (out.get("why") or ""), out)


def r6():
    if not have(REF / "nb0_reference.json", "R6 ref_load negative"):
        return
    if lab() is None:
        skip("R6 ref_load negative", "no lab copy with the export's K100_LAB_SHA")
        return
    pin0 = ref_ns(REF)["K100_PIN"]

    def copy(tag):
        """A copy of the dataset folder (hard links where the file system allows; a changed file is rewritten, not edited)."""
        d = TMP / f"ref_{tag}"
        d.mkdir()
        for f in REF.iterdir():
            if f.is_file():
                try:
                    os.link(f, d / f.name)
                except OSError:
                    shutil.copyfile(f, d / f.name)
        return d

    def rewrite(p, data):
        p.unlink()
        p.write_bytes(data)

    d = copy("trunc")
    p = d / "A_8k_fresh.npz"
    rewrite(p, p.read_bytes()[: p.stat().st_size // 2])
    out = ref_ns(d)["ref_load"]()
    check("R6 one npz truncated (A_8k_fresh half) -> ok False, bad 'sha256 A_8k_fresh.npz', a why",
          out.get("ok") is False and "sha256 A_8k_fresh.npz" in (out.get("bad") or []) and out.get("why"), out)
    d = copy("del")
    (d / "C_40k_repeat.npz").unlink()
    out = ref_ns(d)["ref_load"]()
    check("R6 one npz deleted (C_40k_repeat) -> ok False, bad 'missing C_40k_repeat.npz', a why",
          out.get("ok") is False and "missing C_40k_repeat.npz" in (out.get("bad") or []) and out.get("why"), out)
    pin = json.loads(json.dumps(pin0))
    pin["ref_sha256"]["A_90k_repeat.npz"] = "0" * 64
    out = ref_ns(REF, pin)["ref_load"]()
    check("R6 SHA256SUMS mismatch (the build-time pin of A_90k_repeat differs) -> ok False, bad names it, a why",
          out.get("ok") is False and "sha256 A_90k_repeat.npz" in (out.get("bad") or []) and out.get("why"), out)
    d = copy("ids")
    meta = json.loads((d / "nb0_reference.json").read_text())
    meta["tf"]["A"]["2k"]["prompt_sha256"] = "f" * 64
    rewrite(d / "nb0_reference.json", json.dumps(meta).encode())
    pin = json.loads(json.dumps(pin0))
    pin["ref_sha256"]["nb0_reference.json"] = sha((d / "nb0_reference.json").read_bytes())
    out = ref_ns(d, pin)["ref_load"]()
    check("R6 ids sha != the recorded prompt sha (json re-pinned) -> ok False, bad 'ids A_2k_fresh.npz' / 'ids A_2k_repeat.npz'",
          out.get("ok") is False and {"ids A_2k_fresh.npz", "ids A_2k_repeat.npz"} <= set(out.get("bad") or []) and out.get("why"), out)


# ======================================================================================== R7 / R8 AB + SC reproduction via k100_analysis
def tf_rec(key, band, run_, arr, ns):
    return {"meta": {"boot": key, "band": band, "run": run_, "prompt_sha256": ns["ids_sha"](arr["ids"]), "error": None}, "arr": arr}


def analysis(k0_too=False, scr_k=None):
    ns = ref_ns(REF)
    ns["K1"]["reference"] = ns["ref_load"]()
    ns["K1"]["prompt_identity"] = {}
    for b in BANDS:
        for run_ in ("fresh", "repeat"):
            arr = npz(NB0_RAW / f"B_{b}_{run_}.npz")
            ns["TFR"].setdefault("K", {}).setdefault(b, {})[run_] = tf_rec("K", b, run_, arr, ns)
            if k0_too and run_ == "fresh":
                ns["TFR"].setdefault("K0", {}).setdefault(b, {})[run_] = tf_rec("K0", b, run_, arr, ns)
    if scr_k:
        ns["SCR"]["K"] = scr_k
    return ns, ns["k100_analysis"]()


MET = ("mean_abs_dlogprob", "p99_abs_dlogprob", "rank1_agreement")


def r7():
    if not have(REF / "nb0_reference.json", "R7 AB reproduction"):
        return
    if not (NB0_RAW / "B_90k_repeat.npz").is_file():
        skip("R7 AB reproduction", f"NB0's B arrays (FP8 KV, MTP off) not local: {NB0_RAW}")
        return
    if lab() is None:
        skip("R7 AB reproduction", "no lab copy with the export's K100_LAB_SHA")
        return
    Y = ref_meta()["yardstick"]
    yAB, pB = Y["pairs"]["AB"]["bands"], Y["per_boot"]["B"]
    ns, an = analysis()
    check("R7 k100_analysis runs on the real reference + B as K; reference ok", an["reference"]["ok"] is True, an["reference"])
    for b in BANDS:
        e = an["tf_k_vs_A"][b]
        yf, yl = yAB[b]["full"], yAB[b]["last256"]
        check(f"R7 {b} K(=B) vs A full: mean / p99 / rank-1 = NB0's AB ({yf['mean_abs_dlogprob']} / {yf['p99_abs_dlogprob']} / "
              f"{yf['rank1_agreement']}), n = expected", all(e["full"].get(k) == yf[k] for k in MET + ("max_abs_dlogprob", "n", "expected")),
              ({k: e["full"].get(k) for k in MET + ("n",)}, yf))
        check(f"R7 {b} K(=B) vs A last256: mean / p99 / rank-1 = NB0's AB", all(e["last256"].get(k) == yl[k] for k in MET + ("max_abs_dlogprob",)),
              ({k: e["last256"].get(k) for k in MET}, yl))
        ys = e.get("yardstick_full") or {}
        yl2 = e.get("yardstick_last256") or {}
        check(f"R7 {b} yardstick vs NB0 FP8 KV: mean ratio 1.0, rank-1 delta 0, MIXED (full and last256)",
              ys.get("mean_ratio") == 1.0 and ys.get("rank1_delta") == 0 and ys.get("vs_nb0_fp8_kv") == "MIXED"
              and yl2.get("mean_ratio") == 1.0 and yl2.get("rank1_delta") == 0, (ys, yl2))
        want = KL.tf_label(dict(yf, status="MEASURED"))
        check(f"R7 {b} label_full = the label of NB0's recorded AB ({want}); shift best 0, no artefact",
              e["label_full"] == want and e["shift"]["best_shift"] == 0 and not e["shift"]["alignment_artefact"], (e["label_full"], e["shift"]))
        nf = e["noise_floor"]
        check(f"R7 {b} K's own A/A (B fresh vs repeat) = NB0's recorded B A/A (full + last256)",
              all(nf["k_aa_full"].get(k) == pB["aa_full"][b][k] for k in MET + ("n",))
              and all(nf["k_aa_last256"].get(k) == pB["aa"][b][k] for k in MET), (nf["k_aa_full"], pB["aa_full"][b]))
        check(f"R7 {b} headline = K vs A (no K0)", an["tf_headline"][b]["headline_source"].startswith("K_vs_A")
              and an["tf_headline"][b]["full"] == e["full"])
    want_r = KL.tf_reading({b: {"label_full": KL.tf_label(dict(yAB[b]["full"], status="MEASURED"))} for b in BANDS})
    tr = an["tf_reading"]
    check(f"R7 tf_reading = {want_r['reading']} (worst {want_r.get('worst_band')}), complete",
          tr["reading"] == want_r["reading"] == "LOWBIT_TF_DIVERGENT_SEVERE" and tr["complete"] and tr.get("worst_band") == want_r.get("worst_band"), tr)
    check("R7 K3 / K2 absent -> INCONCLUSIVE_MISSING_TF (never a number)",
          all(v["status"] == "INCONCLUSIVE_MISSING_TF" for v in an["tf_k3_vs_A"].values())
          and an["tf_k2_vs_C_40k"]["status"] == "INCONCLUSIVE_MISSING_TF" and an["tf_k3_vs_A_reading"]["reading"] == "TF_NOT_MEASURED")
    _, an2 = analysis(k0_too=True)
    check("R7 K0 = B too: K0 vs K = 0 / 1.0, headline from K0 (weights only) with the same AB numbers, 90k cross-boot 0",
          all(an2["tf_k0_vs_k"][b]["full"]["mean_abs_dlogprob"] == 0 and an2["tf_k0_vs_k"][b]["full"]["rank1_agreement"] == 1.0
              and an2["tf_headline"][b]["headline_source"].startswith("K0_vs_A")
              and all(an2["tf_headline"][b]["full"][k] == yAB[b]["full"][k] for k in MET) for b in BANDS)
          and an2["tf_k0_vs_k_90k_cross_boot"]["mean_abs_dlogprob"] == 0, {b: an2["tf_k0_vs_k"][b]["full"].get("mean_abs_dlogprob") for b in BANDS})


def r8():
    if not have(REF / "nb0_reference.json", "R8 SC reproduction"):
        return
    if not all((NB0_RAW / f"B_{x}.npz").is_file() for x in ("dec2k_cold", "dec2k_cached", "sc2k_tf", "90k_repeat")):
        skip("R8 SC reproduction", f"NB0's B decode / SC arrays not local: {NB0_RAW}")
        return
    if lab() is None:
        skip("R8 SC reproduction", "no lab copy with the export's K100_LAB_SHA")
        return
    pB = ref_meta()["yardstick"]["per_boot"]["B"]["sc"]
    scr = {}
    for b in ("2k", "8k", "40k"):
        cold, cached, tfz = (npz(NB0_RAW / f"B_dec{b}_cold.npz"), npz(NB0_RAW / f"B_dec{b}_cached.npz"), npz(NB0_RAW / f"B_sc{b}_tf.npz"))
        fake = FakeHTTP()
        ns = tf_ns(fake)
        parse_tf = FULL or b != "40k"
        fake.queue += [json.dumps(dec_response(cold)).encode(), json.dumps(dec_response(cached)).encode()]
        if parse_tf:
            fake.queue.append(json.dumps(tf_response(tfz)).encode())
        ids = [int(x) for x in cold["ids"]]
        dc = ns["decode"]("K", 8001, ids, len(cold["tok"]), b, "cold")
        dd = ns["decode"]("K", 8001, ids, len(cached["tok"]), b, "cached")
        if parse_tf:
            tf = ns["tf_request"]("K", 8001, ids + list(dc["tok"]), "sc" + b, "tf")
        else:                                  # the 40k decode-TF response is about 6 s to build / parse: K100_RD_FULL=1
            tf = tf_rec("K", "sc" + b, "tf", tfz, ns)
        check(f"R8 {b}: B decode{' + decode-TF' if parse_tf else ''} through decode(){' / tf_request()' if parse_tf else ''} "
              f"parse cleanly; the TF arrays = the npz",
              not dc["meta"]["error"] and not dd["meta"]["error"] and tf["arr"] is not None and not same_arrays(tf["arr"], tfz),
              (dc["meta"]["error"], dd["meta"]["error"], tf["meta"].get("error")))
        scr[b] = {"cold": dc, "cached": dd, "tf": tf, "P": len(ids)}
    _, an = analysis(scr_k=scr)
    for b in ("2k", "8k", "40k"):
        got, want = an["self_consistency"][b].get("k_decode_vs_tf") or {}, pB[b]
        keys = ("status", "n", "mean_abs_dlogprob", "p99_abs_dlogprob", "max_abs_dlogprob", "rank1_agreement", "greedy_self_top1",
                "shift_top1_agree", "screen")
        check(f"R8 {b}: k100_analysis k_decode_vs_tf (B via the parse path) = NB0's recorded B decode_vs_tf",
              all(got.get(k) == want.get(k) for k in keys), ({k: got.get(k) for k in keys}, {k: want.get(k) for k in keys}))
        g = an["self_consistency"][b]
        check(f"R8 {b}: greedy_vs_A / cold_vs_cached measured on the same prompt (not INCONCLUSIVE)",
              g["greedy_vs_A"].get("status") in ("EXACT", "NEAR_TIE", "FAIL_HIGH_MARGIN") and
              g["k_cold_vs_cached"].get("status") in ("EXACT", "NEAR_TIE", "FAIL_HIGH_MARGIN"), (g["greedy_vs_A"], g["k_cold_vs_cached"]))


for n, f in (("R1", r1), ("R2", r2), ("R3", r3), ("R4", r4), ("R5", r5), ("R6", r6), ("R7", r7), ("R8", r8)):
    run(n, f)
shutil.rmtree(TMP, ignore_errors=True)
ok = sum(1 for _, v in RES if v)
print(f"{ok}/{len(RES)} PASS", flush=True)
print(f"{len(SKIPS)} skipped" + (": " + "; ".join(f"{n} ({w})" for n, w in SKIPS) if SKIPS else ""), flush=True)
sys.exit(0 if ok == len(RES) else 1)
