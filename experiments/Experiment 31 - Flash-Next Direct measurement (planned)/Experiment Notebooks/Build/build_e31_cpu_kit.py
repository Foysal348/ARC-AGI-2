"""E31 FN DIRECT30 Kaggle CPU kits (esfersami50: the klee100 inputs are shared with that account), the notebook's pinned Py3.12
image, no GPU, no internet, private. Built as a wrapper over the E28 K100 kit builder (Exp 28 Build/build_e28_k100_cpu_kit.py,
exec of its source with exact-once retargeting edits, never written; it wraps the K100Q / NB6 kits): its HEAD, the e2e harness
(the notebook's own cells in one namespace against the fake server, the real Duck payload, the klee100 path on the real
mounts) and the real kit (the real Merlin install, the klee100 merge, every boot plan through Merlin's build(), argparse, the
method map, Q-defer, I8, CELL 0 vs E18 R2) run on the E31 notebook; only the marked E31 edits change them. NOTHING is pushed.

  units       e31_unit_tests.py on Py3.12 (provenance, the E31_CFG line, prompt bytes = ORDER2, the real klee100 tokenizer render
              and STOP_IDS, the multi-test parser regression set, the DIRECT-only adapter + the notebook's CELL 7 scorer + the E15B
              writer, the payload's REAL arc2_duck.py contrast, the solver against an in-process token-id streaming fake: early
              return, forced close token-exact, continue, EOS strip, drain of 24 streams, hard abort, admission stop, request
              errors) + the tf-ref dataset version / payload / pinned-file checks
  e2e-commit  E31DIRECT30 (the 30 tasks, harness E31_CFG: think 6,000 / answer 1,024 / window 50 / drain 3) against the fake server
              EXTENDED for token-id direct streaming (24 parallel streams, never-closing thinking -> forced close, partial and
              labeled multi-test grids, a server-kill injection -> one E15B watchdog relaunch); the real E15B writer, CELL 7, the
              E28 report, the E31 report, CELL 0's fallback, the deadman
  real        the E28 real kit on this notebook; the E28 scheduler check retargeted (e28_sched off: Duck's arc2_duck.py stays the
              manifest file in the commit and the rerun namespace); + E31: CELL 7b on the real merged klee100 tokenizer (render,
              STOP_IDS, EOS from generation_config.json) after CELL 8's own NB5_LOCAL_TOK statement, and the direct request
              body through the INSTALLED Merlin vLLM 0.32 CompletionRequest -> SamplingParams (every field reaches the engine)
Rules: no Kaggle API calls, no git, no local whole-notebook runs (long tests on Kaggle CPU kits only).
Run: python build_e31_cpu_kit.py -> D:/ARC AGI 2/_scratch/push_e25/q27-e31-fn-direct30-kit-<name>__esfersami50/
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E28B = Path(r"D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Build")
E28_KIT = E28B / "build_e28_k100_cpu_kit.py"
E28_KIT_SHA = "994d516fab87e42126073e3af3bb98b1348a13112a2051b4f0fca06ed86d839b"
E31_FACTS = HERE / "e31_build_facts.json"
SLUG, OWNER = "q27-e31-fn-direct30-kit", "esfersami50"
KIT_FACTS = HERE / "e31_kit_facts.json"
HARNESS_CFG = {"think": 6000, "answer": 1024, "window_min": 50, "drain_min": 3, "workers": 24}


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def once(text, old, new, what, count=1):
    assert text.count(old) == count, (what, text.count(old))
    return text.replace(old, new)


def cut(text, a, b=None):
    i = text.index(a)
    return text[i:text.index(b, i)] if b else text[i:]


# ------------------------------------------------------------------ the E28 K100 kit builder, retargeted (exec of its source)
_src = E28_KIT.read_text(encoding="utf-8")
assert sha(_src) == E28_KIT_SHA and "\r" not in _src
_src = once(_src, 'E28_FACTS = HERE / "e28k100_build_facts.json"\n', f'E28_FACTS = Path(r"{E31_FACTS}")   # E31\n', "facts")
_src = once(_src, 'SLUG, OWNER = "q27-e28-k100-kit", "esfersami50"\n', f'SLUG, OWNER = "{SLUG}", "{OWNER}"   # E31\n', "slug")
_src = once(_src, 'assert NB_SHA == FACTS["notebook_sha256"] and FACTS["default_profile"] == NS["COMMIT_PROFILE"] == "E28EVAL120"\n',
            'assert NB_SHA == FACTS["notebook_sha256"] and FACTS["default_profile"] == NS["COMMIT_PROFILE"] == "E31DIRECT30"   # E31\n', "profile")
_src = once(_src, 'assert NS["PHASE"] == FACTS["phase"] == "Q_E28_K100_EVAL120" and NS["DEADMAN_SEC"] == FACTS["deadman_sec"] == 24900\n',
            'assert NS["PHASE"] == FACTS["phase"] == "Q_E31_FN_DIRECT30" and NS["DEADMAN_SEC"] == FACTS["deadman_sec"] == 12300   # E31\n', "phase")
_src = once(_src, 'assert {n: sha((HERE / n).read_bytes()) for n in MODS} == MODS, "the Build modules != the ones packed in the notebook (rebuild)"\n',
            'assert {n: sha(Path(FACTS["module_paths"][n]).read_bytes()) for n in MODS} == MODS, "the Build modules != the packed ones (rebuild)"   # E31\n',
            "modules")
NS2 = {"__file__": str(E28_KIT), "__name__": "e31kit_e28"}
exec(compile(_src, str(E28_KIT), "exec"), NS2)
NS = NS2["NS"]
FACTS = NS2["FACTS"]
CODE = NS2["CODE"]
NB_TXT, NB_SHA = NS2["NB_TXT"], NS2["NB_SHA"]
assert FACTS["e31"]["cfg_line"] in CODE[1] and CODE[6].startswith("# ==================================================================== CELL 7b")

# ------------------------------------------------------------------ HEAD (renamed)
HEAD = NS["HEAD"]
HEAD = once(HEAD, "E28 K100 CPU kit", "E31 FN DIRECT30 CPU kit", "titles", 2)
HEAD = once(HEAD, '"/tmp/e28k100kit"', '"/tmp/e31kit"', "tmp dir")
HEAD = once(HEAD, 'f"e28k100_kit_{KIT}_report.json"', 'f"e31_kit_{KIT}_report.json"', "report name")

# ------------------------------------------------------------------ KF
KF = dict(NS["KF"])
KF.update(e31_modules=FACTS["modules"], e31_cfg=FACTS["e31"]["cfg"], tok_sha="06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
          tpl_sha="c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041", e31_stop_ids=[271, 79939, 279, 6973, 854, 539, 279, 1156,
          11, 353, 599, 310, 2873, 279, 6093, 3018, 383, 279, 7047, 5774, 1381, 13, 198, 248069, 271], harness_cfg=HARNESS_CFG,
          e31_tasks=FACTS["e31"]["tasks"])
UT = (HERE / "e31_unit_tests.py").read_text(encoding="utf-8")
assert "\r" not in UT
KF["e31_unit_checks_min"] = 70
NS["KF"] = KF

# ------------------------------------------------------------------ units (NEW body: the E31 unit tests on Py3.12)
UNITS = r'''
U = {"dataset": dataset_version_check()}
REF = ref_mount()
PC = payload_check(REF) if REF is not None else {"ok": False, "error": "tf-ref dataset not mounted"}
PC.pop("_payload", None)
U["payload"] = PC
U["ds_pins"] = {n: {"pin": h, "ok": REF is not None and (REF / n).is_file() and sha_file(REF / n) == h} for n, h in KF["ds_pins"].items()}
REP["checks"]["units"] = U; save()
B = K / "build"; B.mkdir()
for n, v in FILES.items():
    (B / n).parent.mkdir(parents=True, exist_ok=True)
    (B / n).write_text(v, encoding="utf-8", newline="\n")
# the payload's REAL arc2_duck.py (pre sha b9f67010), from the mounted q27-e15b-payload zip (the notebook's own source)
PAY = find_payload()
U["payload_duck"] = {"used": PAY.get("used")}
try:
    import zipfile as _zf
    with _zf.ZipFile(PAY["used"]) as z_:
        _duck = z_.read("arc2_duck.py")
    (B / "arc2_duck_original.py").write_bytes(_duck)
    U["payload_duck"].update(sha256=hashlib.sha256(_duck).hexdigest(), pre_ok=hashlib.sha256(_duck).hexdigest() == KF["sched_pre"])
except BaseException as e:
    U["payload_duck"].update(pre_ok=False, error=f"{type(e).__name__}: {e}"[:400])
# the klee100 tokenizer from the mounted klee100 Model (sha-pinned = the notebook's TOK_SHA / TPL_SHA)
TK = {"candidates": []}
TOKD = B / "tok"; TOKD.mkdir()
for p_ in sorted(Path(INPUT_ROOT).glob("models/foysalemonshanto/qwen38-flash-next-klee100-3bpw-mtp/*/*/*/tokenizer.json")):
    TK["candidates"].append(str(p_))
    if sha_file(p_) == KF["tok_sha"] and sha_file(p_.parent / "chat_template.jinja") == KF["tpl_sha"]:
        for n_ in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
            shutil.copyfile(p_.parent / n_, TOKD / n_)
        TK["used"] = str(p_.parent)
        break
TK["ok"] = bool(TK.get("used"))
U["tokenizer"] = TK
for k_ in [k_ for k_ in os.environ if k_.startswith(("NB5", "NB6", "E12_", "E15B_", "Q27X_", "RC2_", "E23_", "FAKE_", "K100", "E28", "E31"))] + [
        "E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN"]:
    os.environ.pop(k_, None)
T = {}
try:
    r = sh("e31_unit_tests", [sys.executable, str(B / "e31_unit_tests.py")],
           {"E31T_NB": str(B / "e26.ipynb"), "E31T_FACTS": str(B / "e31_build_facts.json"), "E31T_TOKDIR": str(TOKD),
            "E31T_EVAL": str(COMPD), "E31T_DUCK": str(B / "arc2_duck_original.py"), "E31T_E10_NB": str(B / "no_e10_here"),
            "E31T_EXPORT": str(B / "no_export_here"), "E31T_MERLIN_SRC": str(B / "no_merlin_here"), "PYTHONIOENCODING": "utf-8"}, 1800, cwd=B)
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\d+)/(\d+) PASS", l) for l in lines[::-1] if re.match(r"\d+/\d+ PASS", l)), None)
    T.update(rc=r["rc"], sec=r["sec"], summary=m.group(0) if m else None, fails=[l[:600] for l in lines if l.startswith("FAIL ")][:60],
             skips=[l[:300] for l in lines if l.startswith("SKIP ")], tail=r["tail"][-1500:])
    T["all_pass"] = bool(m) and m.group(1) == m.group(2) and r["rc"] == 0 and not T["fails"] and int(m.group(2)) >= KF["e31_unit_checks_min"]
    # the allowed skips here: the E10 notebook provenance (the ORDER2 fixture is embedded) and the Merlin source (the real kit)
    T["skips_other"] = [l for l in T["skips"] if not l.startswith(("SKIP T3 ORDER2 fixture vs the E10 notebook", "SKIP T8 Merlin source"))]
except BaseException as e:
    T.update(all_pass=False, fatal=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
U["unit_tests"] = T
REP["checks"]["units"] = U; save()
log("e31 unit tests", json.dumps({k: v for k, v in T.items() if k != "tb"})[:3000])
DSV = U["dataset"]
verdict({"tf_ref_k100q_version_mounted": DSV.get("new_version_mounted") is True,
         "tf_ref_every_staged_file_sha_ok": DSV.get("ok") is True,
         "payload_sha_entries_blobs_record_ok": PC.get("ok") is True,
         "dataset_pinned_files_sha_eq_notebook_pins": bool(U["ds_pins"]) and all(v["ok"] for v in U["ds_pins"].values()),
         "duck_payload_arc2_duck_pre_sha": U["payload_duck"].get("pre_ok") is True,
         "klee100_tokenizer_mounted_sha_pinned": TK["ok"] is True,
         "e31_unit_tests_all_pass": T.get("all_pass") is True, "e31_unit_tests_only_allowed_skips": T.get("skips_other") == []},
        recorded={"summary": T.get("summary"), "fails": T.get("fails"), "skips": T.get("skips"), "prereq": DSV.get("prereq"),
                  "tokenizer": TK, "dataset_dir": DSV.get("dir")})
'''

# ------------------------------------------------------------------ the fake server: + E31 token-id direct streaming (extension)
FAKE_SRC = NS["K5"]["src"]("nb5_fake_server.py")
FAKE_EXT = r'''

# ---- E31 (kit extension of the proven fake): token-id DIRECT streaming for the e31 solver. A direct prompt (ORDER2's opening
# sentence) is answered from the fake's eval gold by a per-task script; phase 1 / phase 2 by the prompt (the generated part after
# the template's '<think>\n'); never-closing thinking runs to max_tokens (finish length); a forced-close phase 2 (prompt ends with
# the stop text) gets the answer; a server kill after FAKE_E31_DIE_AFTER_STREAMS direct streams on the FIRST launch only.
E31_HEAD = enc("You are solving an abstract reasoning puzzle.")
E31_STOP = enc("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n")
E31_GEN_MARK = "<|im_start|>assistant\n<think>\n"
E31_FILL = enc("Let me look at the examples and the rule once more. ")
E31_CLOSE = enc("</think>")
E31_EOS = enc("<|im_end|>")
E31 = {"streams": 0, "running": 0, "max_running": 0, "p1": {}, "log": [], "killed": False}


def _e31_rg(g):
    return "\n".join(" ".join(str(int(v)) for v in r) for r in g)


def _e31_task(text):
    for tid, t in CH.items():
        if _e31_rg(t["train"][0]["input"]) in text and all(_e31_rg(q["input"]) in text for q in t["test"]):
            return tid
    return None


def _e31_kind(tid):
    return int(hashlib.sha256(str(tid).encode()).hexdigest(), 16) % 4      # 0 natural / 1 never closes / 2 partial / 3 no grid


def _e31_answer(tid, kind):
    t, gold = CH.get(tid) or {}, SOL.get(tid) or []
    n = len(t.get("test") or [])
    if kind == 3:
        return "\n\nI could not decide on a grid."
    grids = list(gold) if kind in (0, 1) else ([gold[0]] if gold else [])
    if n == 1:
        g = grids[0] if (grids and kind != 2) else [[0]]
        return "\n\n```grid\n" + _e31_rg(g) + "\n```"
    return "\n\n" + "".join(f"```grid {k + 1}\n{_e31_rg(g)}\n```\n" for k, g in reversed(list(enumerate(grids))))   # labels reversed


def e31_direct_stream(handler, body, ids):
    text = dec(ids)
    i = text.find(E31_GEN_MARK)
    gen_text = text[i + len(E31_GEN_MARK):] if i >= 0 else ""
    tid = _e31_task(text[:i] if i >= 0 else text)
    kind = _e31_kind(tid)
    mt = int(body.get("max_tokens", 16))
    # the phase by the solver's seed (SEED + 10 * index + phase): a token-exact phase-1 RESEND (prompt + kept ids) is phase 1
    phase = 1 if (int(body.get("seed") or 0) - 20261005) % 10 == 1 else 2
    rec_resend = phase == 1 and bool(gen_text)
    forced = phase == 2 and ids[-len(E31_STOP):] == E31_STOP
    rec = {"t": time.time(), "port": PORT, "tid": tid, "kind": kind, "phase": phase, "forced": forced, "resend": rec_resend, "prompt_len": len(ids),
           "max_tokens": mt, **{k: body.get(k) for k in ("temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty",
                                                        "frequency_penalty", "repetition_penalty", "skip_special_tokens",
                                                        "return_token_ids", "stream", "ignore_eos", "n", "model")}}
    with LOCK:
        if phase == 1:
            E31["p1"][tid] = {"prompt": list(ids), "sent": []}
        elif forced and tid in E31["p1"]:
            p = E31["p1"][tid]
            rec["prefix_exact"] = list(ids) == p["prompt"] + p["sent"] + E31_STOP
            if STATE_DIR:                    # written at the request START (a later server kill cannot lose it)
                with open(os.path.join(STATE_DIR, "e31_prefix.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": time.time(), "tid": tid, "prefix_exact": rec["prefix_exact"], "pid": os.getpid()}) + "\n")
        E31["streams"] += 1
        n_streams = E31["streams"]
    if phase == 1:
        if kind == 1:
            out, fin = None, "length"
        else:
            out = (E31_FILL * 150 + E31_CLOSE + enc(_e31_answer(tid, kind)) + E31_EOS)   # ~1,650 thinking tokens: 24 streams overlap
            fin = "stop"
    else:
        out = enc(_e31_answer(tid, 0 if kind == 1 else kind) if forced else "```grid\n0\n```") + E31_EOS
        fin = "stop"
    die = int(os.environ.get("FAKE_E31_DIE_AFTER_STREAMS", "0") or 0)
    marker = os.path.join(STATE_DIR, "e31_killed") if STATE_DIR else ""
    rid = f"e31{time.time_ns()}"
    with LOCK:
        WAIT.append(rid)
    while True:
        with LOCK:
            if WAIT and WAIT[0] == rid and len(RUN) < SEQS:
                WAIT.pop(0)
                RUN[rid] = len(ids)
                E31["running"] = sum(1 for k in RUN if k.startswith("e31"))
                E31["max_running"] = max(E31["max_running"], E31["running"])
                break
        time.sleep(0.005)
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Transfer-Encoding", "chunked")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.close_connection = True
    sent = 0
    try:
        first = True
        total = mt if out is None else min(mt, len(out))
        fin_ = "length" if (out is None or len(out) > mt) else fin
        while sent < total:
            n = min(8, total - sent)
            chunk = (E31_FILL * (1 + (sent + n) // len(E31_FILL)))[sent:sent + n] if out is None else out[sent:sent + n]
            sent += n
            d = {"choices": [{"index": 0, "text": dec(chunk), "token_ids": chunk, "finish_reason": fin_ if sent >= total else None}]}
            if first:
                d["choices"][0]["prompt_token_ids"] = list(ids)
            b = b"data: " + json.dumps(d).encode() + b"\n\n"
            handler.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
            handler.wfile.flush()
            with LOCK:
                if first:
                    account(ids, cached_len(ids), 0)
                    first = False
                ST["gen"] += n
                WIN["gen"] += n
                RUN[rid] = RUN.get(rid, 0) + n
                if phase == 1 and tid in E31["p1"]:
                    E31["p1"][tid]["sent"].extend(chunk)
            if die and n_streams >= die and marker and not os.path.exists(marker):
                open(marker, "w").write(str(time.time()))
                launch_record("killed", e31_streams=n_streams)
                print("fake merlin: E31 server-kill injection", flush=True)
                os._exit(1)
            time.sleep(n / SPEED)
        b = b"data: [DONE]\n\n"
        handler.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n" + b"0\r\n\r\n")
        handler.wfile.flush()
        rec["finish"] = fin_
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
        rec["finish"] = "client_closed"
    finally:
        rec.update(sent=sent, t_end=time.time())
        with LOCK:
            RUN.pop(rid, None)
            if rid in WAIT:
                WAIT.remove(rid)
            E31["running"] = sum(1 for k in RUN if k.startswith("e31"))
        if STATE_DIR:
            with open(os.path.join(STATE_DIR, "e31_requests.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            _sp = os.path.join(STATE_DIR, f"e31_stats_{os.getpid()}.json")
            with LOCK:
                with open(_sp + ".tmp", "w", encoding="utf-8") as f:
                    json.dump({k: E31[k] for k in ("streams", "max_running", "killed")}, f)
                os.replace(_sp + ".tmp", _sp)
'''
FAKE_E31 = once(FAKE_SRC, '    ids = list(body.get("prompt") or [])\n    max_t = int(body.get("max_tokens", 16))\n    rid = f"s{time.time_ns()}"\n',
                '    ids = list(body.get("prompt") or [])\n'
                '    if any(ids[i:i + len(E31_HEAD)] == E31_HEAD for i in range(min(len(ids), 240))):   # E31: a direct request\n'
                '        return e31_direct_stream(handler, body, ids)\n'
                '    max_t = int(body.get("max_tokens", 16))\n    rid = f"s{time.time_ns()}"\n', "fake hook")
FAKE_E31 = once(FAKE_E31, "\n\ndef stats_json():\n", FAKE_EXT + "\n\ndef stats_json():\n", "fake ext")
compile(FAKE_E31, "<e31 fake>", "exec")

# ------------------------------------------------------------------ the E31 e2e driver (the E28 kit's harness + E31 checks)
D = NS2["E2E_DRV"]
A = D[:D.index('want_profile = "RERUN_PROD" if RR else "')]
A = A.replace('"""E26 K100 QUALITY e2e', '"""E31 FN DIRECT30 e2e (= the E28 K100 kit\'s driver: E26 K100 QUALITY e2e', 1)
A = once(A, 'else:      # E28: the commit too gets a FAKE SHORT deadline (the profile is 415 min); the harness B2 budget\n'
            '    os.environ["Q27X_TIME_BUDGET"] = str(BUDGET)\n',
         'else:      # E31: the commit runs its OWN profile budget from the harness E31_CFG (window 50 -> 85 min) + a server kill\n'
         f'    os.environ.update(E31_CFG_JSON=json.dumps({json.dumps(HARNESS_CFG)}), FAKE_E31_DIE_AFTER_STREAMS="33", E15B_WD_MIN_LEFT_SEC="300")\n',
         "commit env")
assert "Q27X_TIME_BUDGET=str(BUDGET)" in A
B1 = r'''want_profile = "RERUN_PROD" if RR else "E31DIRECT30"   # E31
check("every code cell ran without an uncaught exception", cell_err is None, cell_err)
check("E31 layout: 9 code cells (CELL 0, 1, 3, order, 5, 6/7, 7b, 8, 9)", len(code) == 9 and "CELL 7b" in heads[6], heads)
check(f"PROFILE == {want_profile} in the notebook, CONFIG and the report; RERUN == {RR}",
      lambda: G["PROFILE"] == want_profile and G["CONFIG"]["PROFILE"] == want_profile and rep["profile"] == want_profile
      and G["RERUN"] is (RR) and rep["rerun"] is (RR), (G.get("PROFILE"), rep.get("profile"), G.get("RERUN")))
check("NB5 report has no runner error", lambda: rep and not rep.get("errors"), rep.get("errors"))
want_ph = [p[0] for p in P.get("nb5_phases") or []]
check("phases run = the profile's phases (['Q_E31_FN_DIRECT30']), all DONE",
      lambda: want_ph == ["Q_E31_FN_DIRECT30"] and list(ph) == want_ph and all(ph[n].get("status") == "DONE" for n in want_ph),
      {n: (p.get("status"), (p.get("error") or "")[-600:]) for n, p in ph.items()})
boots = rep.get("boots") or []
b0 = boots[-1] if boots else {}
_bsum = json.dumps([{k: b.get(k) for k in ("tag", "status", "kv_dtype", "mtp")} | {"tries": [(t_.get("pin_gib"), t_.get("status"))
                    for t_ in b.get("tries") or []]} for b in boots], default=str)[:1500]
check("one boot (FP8, MTP on), the first K3 rung booted with the INT8 draft head (E28's levers), probes without a fatal",
      lambda: len(boots) == 1 and b0["kv_dtype"] == "fp8" and b0["mtp"] is True and (b0.get("probes") or {}).get("fatal") is None
      and len(b0["tries"]) == 1 and b0["tries"][0].get("e28_i8") is True and b0["tries"][0]["name"].endswith("_i8"), _bsum)
cap = b0.get("capacity") or {}
check("capacity of the booted try: the top rung, launch max_num_seqs 24, workers 24 (E28's)",
      lambda: abs(b0["pin_used"]["gib"] - __G1__) < 1e-6 and cap["caps"]["max_num_seqs"] == 24 and cap["workers"] == 24, json.dumps(cap, default=str)[:900])
_lv = {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"}
check("lever env MERLIN_SHARE_DRAFT_WORKSPACE=1 reached every (fake) server process that booted (the first and the relaunch)",
      lambda: [L_["env_levers"] for L_ in launches if L_["outcome"] == "ready"] == [_lv] * len([L_ for L_ in launches if L_["outcome"] == "ready"])
      and all(b0["server_env_levers"].get(k_) == v_ for k_, v_ in _lv.items()), json.dumps([(L_.get("outcome"), L_.get("env_levers")) for L_ in launches]))
main = ph.get("Q_E31_FN_DIRECT30") or {}
'''
S = D[D.index("# ---- submission.json\n"):D.index("else:\n    check(\"commit: submission.json = CELL 0's identity fallback")]
B2 = r'''else:
    check("commit: submission.json = CELL 0's identity fallback of the test file (the eval subset never writes it)",
          lambda: sub == {t: ident(t) for t in test}, "")
    sc = main.get("score") or {}
    check("commit (E31): the 30 eval tasks (task_list), 46 test inputs scored by exact match (CELL 7 = the E15B report), >= 1 hit",
          lambda: P["source"] == "eval" and sorted(P["task_list"]) == sorted(KF_TASKS) and sorted(G["TASK_IDS"]) == sorted(KF_TASKS)
          and sc["eval_query_total"] == 46 and sc["eval_query_hits"] >= 1, json.dumps({"score": sc}, default=str)[:900])
    check("commit: the profile budget from the harness E31_CFG: 15 + 50 + 20 = 85 min (5,100 s), no Q27X_TIME_BUDGET override",
          lambda: G["E31_BUDGET_MIN"] == 85 and P["time_budget_sec"] == 5100 and G["TOTAL_BUDGET_SEC"] == 5100 and G["E31_CFG"]["think"] == 6000,
          G.get("TOTAL_BUDGET_SEC"))
'''
KF_TASKS_LINE = f"KF_TASKS = {json.dumps(FACTS['e31']['tasks'])}\n"
T_ = D[D.index("_hard_wall = time.time()"):D.index("\n\n# ---- E26 K100: the klee100 path on the REAL mounts")]
E26 = D[D.index("\n\n# ---- E26 K100: the klee100 path on the REAL mounts"):D.index('KQP = W / "q27x" / "e28_k100_report.json"   # E28\nKQ =')]
E26 = once(E26, '      lambda: len(_la) == 1 and _last(_la[0]["argv"], "--max-num-seqs") == str(_cw[3])',
           '      lambda: len(_la) >= 1 and _last(_la[0]["argv"], "--max-num-seqs") == "24"   # E31: + the relaunch (same argv)', "launch argv")
E26 = once(E26, '      and _last(_la[0]["argv"], "--kv-cache-memory-bytes") == str(6859784192 if RETRY else 7933526016), json.dumps(_la, default=str)[:1500])',
           '      and _last(_la[0]["argv"], "--kv-cache-memory-bytes") == "7933526016" and all(L_["argv"] == _la[0]["argv"] for L_ in _la),\n'
           '      json.dumps(_la, default=str)[:1500])', "launch pin")
B3 = r'''KQP = W / "q27x" / "e28_k100_report.json"   # E28 (CELL 9, unchanged)
KQ = json.loads(KQP.read_text(encoding="utf-8")) if KQP.is_file() else {}
REC["e28_report"] = {k: KQ.get(k) for k in ("status", "error", "errors", "score", "attempt_summary")}
_SR = G.get("E28_SCHED_REC") or {}
check("E31: e28_sched off: CELL 3 left Duck's arc2_duck.py the manifest file (want False, applied False), no E28_SCHED env",
      lambda: _SR == {"want": False, "applied": False} and os.environ.get("E28_SCHED") is None and not hasattr(G["E12A"], "_e28_solve_all"), _SR)
_E31R = G.get("E31_RUN") or {}
check("E31: CELL 7b rebound E12A.solve_all_r6 / build_submission; ONE solver call with 24 workers on SGL_URL (the booted NB5_PORT "
      "server), the Duck-only arguments recorded as ignored",
      lambda: G["E12A"].solve_all_r6 is G["e31_solve_all_r6"] and G["E12A"].build_submission is G["e31_build_submission"]
      and len(_E31R["calls"]) == 1 and _E31R["calls"][0]["workers"] == 24 and _E31R["calls"][0]["base_url"] == G["SGL_URL"] + "/v1"
      and G["SGL_URL"].endswith(":" + os.environ["NB5_PORT"]) and "max_generated_tokens" in _E31R["calls"][0]["ignored"], _E31R)
_ER = json.loads((W / "q27x" / "e31_direct_report.json").read_text(encoding="utf-8")) if (W / "q27x" / "e31_direct_report.json").is_file() else {}
_es = _ER.get("sched") or {}
REC["e31_report"] = {k: _ER.get(k) for k in ("status", "tasks_finished", "modes", "end_reasons", "mapping_sources", "speed", "think_coverage")}
REC["e31_groups"] = _ER.get("groups")
check("E31 report: SCORED, all 30 tasks finished (the window is a cap: the solver returned early), modes natural + forced_close, "
      "no task ended 'error', label + fallback mappings, the speed fields",
      lambda: _ER["status"] == "SCORED" and _ER["tasks_finished"] == 30 and _ER["modes"].get("natural", 0) >= 1
      and _ER["modes"].get("forced_close", 0) >= 1 and "error" not in _ER["end_reasons"] and _ER["mapping_sources"].get("label", 0) >= 1
      and _ER["mapping_sources"].get("fallback", 0) >= 1 and _ER["speed"]["returned_early"] is True and _ER["speed"]["aggregate_decode_tok_s"] > 0
      and _ER["speed"]["solve_wall_sec"] < 50 * 60, REC["e31_report"])
check("E31 report: groups main / partial / control / all + multi_test / single_test with query_hits, fully_solved_tasks, task_norm_pct; "
      "coverage at 60k / 78k / 104k; the per-task comparison vs K100 / W4 / Direct27B / Duck",
      lambda: all(set(("query_hits", "fully_solved_tasks", "task_norm_pct")) <= set(_ER["groups"][g]) for g in ("main", "partial", "control", "all",
                                                                                                             "multi_test", "single_test"))
      and _ER["groups"]["all"]["queries"] == 46 and set(_ER["think_coverage"]) == {"60000", "78000", "104000"}
      and set(_ER["comparison_vs_reference"]) == {"K100", "W4", "Direct27B", "Duck"}, REC["e31_groups"])
_rows = [json.loads(l_) for l_ in (W / "q27x" / "e31_direct_records.jsonl").read_text(encoding="utf-8").splitlines() if l_.strip()] \
    if (W / "q27x" / "e31_direct_records.jsonl").is_file() else []
check("E31 records: 46 per-test rows (unit, query_index, mapping_source, fallback_reason, mode, end_reason, tokens, hit / pass1, group)",
      lambda: len(_rows) == 46 and len({r_["unit"] for r_ in _rows}) == 46 and all(r_["group"] in ("main", "partial", "control")
      and "mapping_source" in r_ and "fallback_reason" in r_ and r_["hit"] in (0, 1) for r_ in _rows), len(_rows))
_PD = W / "q27x" / "phase_Q_E31_FN_DIRECT30"
_ga = json.loads((_PD / "e12_generated_attempts.json").read_text(encoding="utf-8")) if (_PD / "e12_generated_attempts.json").is_file() else {}
_rr = json.loads((_PD / "e12_run_report.json").read_text(encoding="utf-8")) if (_PD / "e12_run_report.json").is_file() else {}
_E12R = G.get("E12_RECORDS") or {}
check("E31: CELL 7's generated attempts == the direct records' own ORDER2 make_attempts (DIRECT-only), one record per task, and "
      "CELL 7's score == the E31 report's query hits",
      lambda: set(_ga) == set(KF_TASKS) and all(_ga[t] == _E12R[t][0]["e31_attempts"] for t in KF_TASKS) and all(len(_E12R[t]) == 1 for t in KF_TASKS)
      and _rr["eval_query_hits"] == _ER["groups"]["all"]["query_hits"] and _rr["eval_query_total"] == 46, {"cell7": _rr.get("eval_query_hits")})
check("E31: the E28 report (CELL 9, unchanged) is VALID on the direct records: its score of the written attempts == CELL 7's",
      lambda: KQ["status"] == "VALID" and KQ["score"]["agree"] is True and len(KQ["attempts"]) == 30, REC["e28_report"])
check("E31: E12A.SCHED_LAST came from the direct scheduler (E12_RUN rev6_scheduler: the e31 section, gauges, 24 workers)",
      lambda: G["E12_RUN"]["rev6_scheduler"]["e31"]["admitted"] == 30 and G["E12_RUN"]["rev6_scheduler"]["workers"] == 24
      and isinstance(G["E12_RUN"]["rev6_scheduler"]["gauges"], list), "")
def _jl(p_):        # a line cut by the server-kill injection is skipped (counted)
    out_ = []
    for l_ in p_.read_text(encoding="utf-8").splitlines():
        try:
            out_.append(json.loads(l_)) if l_.strip() else None
        except ValueError:
            REC["fake_cut_lines"] = REC.get("fake_cut_lines", 0) + 1
    return out_


_fr = [r_ for p_ in sorted((WORK / "fake_state").glob("e31_requests.jsonl")) for r_ in _jl(p_)]
_fs = [s_ for p_ in (WORK / "fake_state").glob("e31_stats_*.json") for s_ in [(lambda q_: (lambda: json.loads(q_.read_text(encoding="utf-8")))())(p_)
        if p_.read_text(encoding="utf-8").strip().endswith("}") else {"max_running": 0}]]
REC["fake_e31"] = {"requests": len(_fr), "stats": _fs, "kinds": sorted({r_["kind"] for r_ in _fr}), "forced": sum(1 for r_ in _fr if r_["forced"]),
                   "prefix_exact": [r_.get("prefix_exact") for p_ in sorted((WORK / "fake_state").glob("e31_prefix.jsonl")) for r_ in _jl(p_)]}
check("E31 fake: every direct request on the booted port with every sampling field explicit (T 0.6, top_p 0.95, top_k 20, min_p 0, "
      "penalties neutral, a seed, stream + return_token_ids, skip_special_tokens False); 24 direct streams ran at once",
      lambda: _fr and all(r_["port"] == int(os.environ["NB5_PORT"]) and r_["temperature"] == 0.6 and r_["top_p"] == 0.95 and r_["top_k"] == 20
                          and r_["min_p"] == 0.0 and r_["presence_penalty"] == 0.0 and r_["frequency_penalty"] == 0.0 and r_["repetition_penalty"] == 1.0
                          and isinstance(r_["seed"], int) and r_["stream"] is True and r_["return_token_ids"] is True
                          and r_["skip_special_tokens"] is False for r_ in _fr) and max(s_["max_running"] for s_ in _fs) == 24, REC["fake_e31"])
check("E31 fake: never-closing thinking -> forced-close phase 2 requests; every forced-close prompt whose phase 1 the same fake process "
      "served == prompt ids + the streamed ids + the 25 stop ids exactly",
      lambda: REC["fake_e31"]["forced"] >= 1 and REC["fake_e31"]["prefix_exact"] and all(REC["fake_e31"]["prefix_exact"]), REC["fake_e31"])
_rl = rep.get("relaunches") or []
_wd = (G.get("E12_RUN") or {}).get("watchdog") or {}
check("E31 server kill: the fake died once mid-solve -> the E15B watchdog relaunched it ONCE (e21_relaunch ok, the same port); the "
      "solver counted the request errors, waited, continued, and no error was reported as a finish",
      lambda: sum(1 for L_ in launches if L_["outcome"] == "killed") == 1 and len([L_ for L_ in launches if L_["outcome"] == "ready"]) == 2
      and len(_rl) == 1 and _rl[0]["ok"] is True and len(_wd.get("relaunches") or []) == 1 and _es["request_errors"] >= 1
      and _es["error_retries"] >= 1 and _es["error_finishes"] == 0, json.dumps({"relaunches": _rl, "wd": _wd.get("relaunches"),
                                                                                 "errors": _es.get("request_errors")}, default=str)[:1500])
check("E31: the full thinking of every request saved (phase duck/<task>_b0/transcript.txt + token_ids.json, 30 tasks) and in the Duck bundle",
      lambda: all((_PD / "duck" / f"{t}_b0" / "transcript.txt").is_file() and (_PD / "duck" / f"{t}_b0" / "token_ids.json").is_file() for t in KF_TASKS)
      and sum(1 for n_ in _zf.ZipFile(next(W.glob("Q27X_E21_*_DUCK_BUNDLE.zip"))).namelist() if n_.endswith("transcript.txt")) == 30, "")
import threading as _th
_DM = G.get("K100_DEADMAN") or {}
_dmt = [t_ for t_ in _th.enumerate() if t_.name == "k100q_deadman"]
_dmf = W / "k100q_deadman.txt"
REC["deadman"] = {"armed": _DM.get("armed"), "fire_in_sec": round(_DM["fire_pc"] - G["NOTEBOOK_T0"], 3) if _DM.get("fire_pc") else None,
                  "threads": [(t_.name, t_.daemon, t_.is_alive()) for t_ in _dmt], "marker_exists": _dmf.exists(), "sec": G.get("K100_DEADMAN_SEC")}
check("commit: the deadman is armed at NOTEBOOK_T0 + E31_BUDGET_MIN * 60 (harness 5,100 s; the GPU build 12,300 s), one daemon thread, "
      "still sleeping at the end (it did NOT fire)",
      lambda: _DM["armed"] is True and G["K100_DEADMAN_SEC"] == G["E31_BUDGET_MIN"] * 60 == 5100 and abs(_DM["fire_pc"] - (G["NOTEBOOK_T0"] + 5100)) < 1e-6
      and len(_dmt) == 1 and _dmt[0].daemon is True and _dmt[0].is_alive() and not _dmf.exists() and T_END < _DM["fire_pc"], REC["deadman"])
_bz = W / "Q27X_NB5_BUNDLE.zip"
check("the NB5 bundle zip carries e28_k100_report.json", lambda: any(n_.endswith("e28_k100_report.json") for n_ in _zf.ZipFile(_bz).namelist()), str(_bz))
'''
F = D[D.index("bad = [c for c in checks if not c[1]]\n"):]
E2E_DRV = A + KF_TASKS_LINE + B1 + S + B2 + T_ + E26 + "\nimport zipfile as _zf\n" + B3 + F
E2E_DRV = E2E_DRV.replace("__G1__", repr(NS["GIBS"][0]))
assert "E28EVAL120" not in E2E_DRV and "__" + "G1__" not in E2E_DRV
compile(E2E_DRV, "<e31_e2e>", "exec")

E2E = NS2["E2E"]
E2E = once(E2E, '    _max = 3500 if mode != "commit" else 4200   # E28: the commit runs on a fake 3,000-s deadline (+ the report / bundle)\n',
           '    _max = 3500 if mode != "commit" else 6600   # E31: the commit runs its harness budget (85 min) + the report / bundle\n', "commit max")
E2E = E2E.replace("__MODE__", "commit")

# ------------------------------------------------------------------ real (the E28 real kit, the scheduler check retargeted, + E31)
REAL = NS2["REAL"]
_s0 = REAL.index('E28R = {}\ntry:\n    S_ = G["E28S"]\n')
_s1 = REAL.index('try:\n    E28R["i8_static"] = G["e28_i8_static"]()\n')
REAL = REAL[:_s0] + r'''E28R = {}
try:      # E31: e28_sched off -> the commit AND the rerun namespace keep Duck's arc2_duck.py = the manifest file (no scheduler)
    E12D = Path(os.environ.get("E12_DIR") or (K / "e12_duck"))
    E28R["sched_commit"] = G.get("E28_SCHED_REC")
    E28R["sched_rerun"] = GR.get("E28_SCHED_REC")
    E28R["module_sha"] = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in (G.get("E28_SRC") or {}).items()}
    E28R["duck_sha"] = hashlib.sha256((E12D / "arc2_duck.py").read_bytes()).hexdigest()
    E28R["sched_ok"] = (E28R["sched_commit"] == {"want": False, "applied": False} and E28R["sched_rerun"] == {"want": False, "applied": False}
                        and E28R["duck_sha"] == G["_E12_MANIFEST"]["arc2_duck.py"] == KF["sched_pre"] and os.environ.get("E28_SCHED") is None
                        and E28R["module_sha"] == KF["e31_modules"] and G.get("E28S") is None)
except BaseException as e:
    E28R.update(sched_ok=False, sched_error=f"{type(e).__name__}: {e}"[:800], sched_tb=traceback.format_exc()[-1500:])
''' + REAL[_s1:]
REAL = once(REAL, '"e28_sched_applied_commit_rerun_revert_reapply": E28R.get("sched_ok") is True,',
            '"e31_duck_original_no_e28_sched_commit_rerun": E28R.get("sched_ok") is True,', "verdict key")
_plat = NS2["_plat"]
PROTO_PROBE = _plat.replace("ext_src, cfg = open(sys.argv[1]).read(), json.load(open(sys.argv[2]))\n", "") + r'''
try:
    body = json.loads(open(sys.argv[1]).read())
    import vllm.entrypoints.openai.completion.protocol as PR
    req = PR.CompletionRequest(**body)
    srv_default = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
    sp = req.to_sampling_params(int(body["max_tokens"]), default_sampling_params=srv_default)
    import inspect
    import vllm.entrypoints.openai.completion.serving as SV
    ssrc = inspect.getsource(SV)
    out["steps"]["proto"] = {"ok": True, "file": PR.__file__, "stream": req.stream, "return_token_ids": req.return_token_ids,
                             "prompt_is_ids": isinstance(req.prompt, list) and all(isinstance(x, int) for x in req.prompt),
                             "sp": {k: (getattr(sp, k, None) if not hasattr(getattr(sp, k, None), "name") else getattr(sp, k).name)
                                    for k in ("temperature", "top_p", "top_k", "min_p", "seed", "max_tokens", "presence_penalty",
                                              "frequency_penalty", "repetition_penalty", "skip_special_tokens", "ignore_eos", "n",
                                              "output_kind")},
                             "serving_streams_token_ids": "token_ids=(" in ssrc and "if request.return_token_ids" in ssrc}
except BaseException as e:
    out["steps"]["proto"] = {"ok": False, "error": err(e), "tb": traceback.format_exc()[-1500:]}
print("@JSON " + json.dumps(out, default=str))
'''
compile(PROTO_PROBE, "<e31 proto probe>", "exec")
sys.path.insert(0, str(HERE))
import e31_direct as E31D   # noqa: E402
PROBE_BODY = E31D.request_body("q27", [1, 2, 3, 4], 104000, 0.6, 20261047)
_c8 = CODE[7]
_tok_stmt = next(ast.get_source_segment(_c8, n) for n in ast.walk(ast.parse(_c8)) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and ast.get_source_segment(_c8, n.value.func) == "NB5_LOCAL_TOK.update")
E31_REAL = ("\nPROTO_PROBE = json.loads(" + repr(json.dumps(PROTO_PROBE)) + ")   # E31: the direct request body through the installed Merlin\n"
            "E31_TOK_STMT = json.loads(" + repr(json.dumps(_tok_stmt)) + ")   # E31: CELL 8's own NB5_LOCAL_TOK statement (verbatim)\n"
            "PROBE_BODY = json.loads(" + repr(json.dumps(PROBE_BODY)) + ")\n") + r'''
# ---------------------------------------------------------------- E31: CELL 7b on the real merged klee100 tokenizer + the request body on the installed Merlin
E31V = {}
try:
    for _i in (5, 6):                                     # CELL 6/7 (defs + the packed e21_cell7) and CELL 7b (the rebind), the notebook's text
        exec(compile(CODE[_i], f"<e31 real cell {_i}>", "exec"), G)
    exec(compile(E31_TOK_STMT, "<CELL 8 NB5_LOCAL_TOK>", "exec"), G)
    _t = G["e31_tok"]()
    _E31D = G["E31D"]
    _tid = sorted(json.load(open(COMPD / "arc-agi_evaluation_challenges.json")))[0]
    _task = json.load(open(COMPD / "arc-agi_evaluation_challenges.json"))[_tid]
    _txt = _t["render"]([{"role": "user", "content": _E31D.task_message(_task)}])
    _ids = _t["encode"](_txt)
    E31V["tok"] = {"tokenizer": G["E31_RUN"].get("tokenizer"), "prompt_tokens": len(_ids), "task": _tid,
                   "head_xhigh": _txt.startswith("<|im_start|>system\nReasoning effort is set to xhigh."),
                   "tail": _txt.endswith("<|im_start|>assistant\n<think>\n"), "ids_tail": _ids[-2:],
                   "stop_ids": _t["encode"](_E31D.STOP_THINKING_TEXT), "eos": _t["eos_ids"], "verified": G["NB5_LOCAL_TOK"].get("verified"),
                   "rebind": G["E12A"].solve_all_r6 is G["e31_solve_all_r6"] and G["E12A"].build_submission is G["e31_build_submission"]}
    E31V["tok_ok"] = (E31V["tok"]["head_xhigh"] and E31V["tok"]["tail"] and E31V["tok"]["ids_tail"] == [248068, 198]
                      and E31V["tok"]["stop_ids"] == KF["e31_stop_ids"] and E31V["tok"]["eos"] == [248046, 248044] and E31V["tok"]["verified"] is True
                      and E31V["tok"]["rebind"] and (E31V["tok"]["tokenizer"] or {}).get("stop_ids_equal_spec") is True)
except BaseException as e:
    E31V.update(tok_ok=False, tok_error=f"{type(e).__name__}: {e}"[:800], tok_tb=traceback.format_exc()[-1500:])
try:
    (K / "e31_probe_body.json").write_text(json.dumps(PROBE_BODY))
    r = venv_py(PROTO_PROBE, [str(K / "e31_probe_body.json")], name="e31_proto_probe", extra_env=dict(globals().get("mm_env") or {"CUDA_VISIBLE_DEVICES": ""}))
    js = r["json"] or {"error": r["tail"]}
    E31V["proto"] = ((js.get("steps") or {}) if isinstance(js, dict) else js).get("proto") if isinstance(js, dict) else js
    _p = E31V["proto"] or {}
    _sp = _p.get("sp") or {}
    E31V["proto_ok"] = (_p.get("ok") is True and _p.get("stream") is True and _p.get("return_token_ids") is True and _p.get("prompt_is_ids") is True
                        and _sp.get("temperature") == 0.6 and _sp.get("top_p") == 0.95 and _sp.get("top_k") == 20 and _sp.get("min_p") == 0.0
                        and _sp.get("seed") == PROBE_BODY["seed"] and _sp.get("max_tokens") == 104000 and _sp.get("skip_special_tokens") is False
                        and _sp.get("ignore_eos") is False and _sp.get("presence_penalty") == 0.0 and _sp.get("frequency_penalty") == 0.0
                        and _sp.get("repetition_penalty") == 1.0 and _p.get("serving_streams_token_ids") is True
                        and str(_p.get("file") or "").startswith(str(Path(G["MERLIN_ENG"]["kv"]["MERLIN_TREE"]))))
except BaseException as e:
    E31V.update(proto_ok=False, proto_error=f"{type(e).__name__}: {e}"[:600])
try:
    E31V["gen_config_in_ckpt"] = json.loads((G["CKPT"] / "generation_config.json").read_text(encoding="utf-8")).get("eos_token_id")
except BaseException as e:
    E31V["gen_config_in_ckpt"] = f"{type(e).__name__}: {e}"[:300]
E28V.update({"e31_cell7b_real_klee100_tokenizer_render_stop_ids_eos": E31V.get("tok_ok") is True,
             "e31_request_body_through_installed_merlin_completion_protocol": E31V.get("proto_ok") is True,
             "e31_ckpt_generation_config_eos": E31V.get("gen_config_in_ckpt") == [248046, 248044]})
REP["checks"]["e31"] = E31V; save()
log("E31 real", json.dumps({k: v for k, v in E31V.items() if not k.endswith("tb")}, default=str)[:4000])
'''
_v = REAL.index('REP["checks"]["e28"] = E28R; save()\n')
REAL = REAL[:_v] + E31_REAL + REAL[_v:]
compile(REAL, "<e31 real>", "exec")

# ------------------------------------------------------------------ the kit table
UNIT_FILES = {"e26.ipynb": NB_TXT, "e31_build_facts.json": E31_FACTS.read_bytes().decode("utf-8"), "e31_unit_tests.py": UT}
E2E_FILES = dict(NS2["E2E_FILES"], **{"e26_e2e.py": E2E_DRV, "nb5_fake_server.py": FAKE_E31})
assert E2E_FILES["e26.ipynb"] == NB_TXT
REAL_FILES = dict(NS2["REAL_FILES"])
assert REAL_FILES["e26.ipynb"] == NB_TXT
NS["HEAD"] = HEAD
NS["KITS"] = {
    "units": {"body": UNITS, "title": "Q27 E31 FN Direct30 Kit Units", "models": NS["MODELS"], "files": UNIT_FILES},
    "e2e-commit": {"body": E2E, "title": "Q27 E31 FN Direct30 Kit E2E Commit", "models": NS["MODELS"], "files": E2E_FILES},
    "real": {"body": REAL, "title": "Q27 E31 FN Direct30 Kit Real", "models": NS["MODELS"], "files": REAL_FILES},
}

if __name__ == "__main__":
    print("notebook under test", NS["NB_NAME"], "sha256", NB_SHA, "| profile", NS["COMMIT_PROFILE"], "| phase", NS["PHASE"], flush=True)
    out = {}
    for n in NS["KITS"]:
        d, h, nbytes = NS["build"](n)
        out[n] = {"folder": str(d), "sha256": h, "bytes": nbytes, "kit_version": "v1"}
    KIT_FACTS.write_bytes((json.dumps({"notebook_sha256": NB_SHA, "e28_kit": str(E28_KIT), "e28_kit_sha256": E28_KIT_SHA, "kits": out,
                                       "kf": {k: KF[k] for k in ("sched_pre", "e31_modules", "e31_unit_checks_min", "harness_cfg")}},
                                      indent=1) + "\n").encode("utf-8"))
