"""E31 Flash-Next DIRECT (user 2026-11-01): the proven Qwen3.8-27B DIRECT method (v1.4.4 / E10 ORDER2, the spec in Exp 31
Experiment Review/Agent Reports/DIRECT_solver_spec_E10_ORDER2_agent_report_2026-11-01.md) on the Merlin vLLM-0.32 OpenAI
server of the E28 K100 boot. Copied, not re-invented:
  - prompt: russcore's free-form prompt = ORDER2 build_user_message, verbatim for a single-test task; ONE prompt per TASK
    (user 2026-11-01 design change): a multi-test task lists "FINAL INPUT k:" blocks and asks for labeled ```grid k blocks
    (every other sentence = ORDER2's);
  - chat template: the checkpoint's own (enable_thinking True, reasoning_effort xhigh), rendered text -> token ids;
  - phase 1 + forced-close phase 2, token-exact (prompt ids + phase-1 ids + STOP_IDS), "continue" when </think> closed but
    the answer was cut; trailing EOS stripped before a forced close (ORDER2 F8);
  - parse_grid / make_attempts verbatim (attempt_1 = the model grid, attempt_2 = the test input or its mirror); a multi-test
    answer: labeled blocks "grid K" -> test K, missing labels -> the last N valid fenced blocks in order, a test with no grid
    -> make_attempts' fallback for that test only;
  - ORDER2 order (score = [>= 3 train] + [1 test], ties in sorted-id order), seeds SEED + 10 * task_index + phase;
  - ORDER2's scheduler loop (continuous admission, drain -> forced close, hard stop) with the audited phases inside the
    solve window T: no new task from T - 12 min, drain at T - drain_min, every request aborted at T - 1 min (partial
    answers parsed), the last minute for writing; on E18's streaming token-id client ONLY (/v1/completions, prompt = ids,
    return_token_ids, abort by closing the stream; no E18 relaunch: the notebook's E15B watchdog owns the server);
  - every sampling field sent explicitly (the server fills a missing one from --override-generation-config: T 1.0).
The solver RETURNS as soon as every task is done (the window is an upper cap, never a wait). No quality / label gate stops
or skips work. Pure module: every notebook object (tokenizer, url, callbacks) is passed in; unit-tested in Build/e31_unit_tests.py.
"""
import json
import queue as _queue
import re
import socket
import threading
import time
import urllib.request as _urlreq
from pathlib import Path

LIB_VERSION = "e31_direct 1.0 (2026-11-01)"

SEED = 20261005                                  # ORDER2 nb[2]
MAX_MODEL_LEN = 131072                           # the E28 K100 server's --max-model-len (unchanged)
CONTEXT_SAFETY = 256                             # ORDER2
MIN_THINK_BUDGET = 4096                          # ORDER2: a unit is skipped (insufficient_context) only below this
THINK_FLOOR = 60000                              # user rule: thinking >= 60k unless the prompt forbids (recorded)
# the phased drain inside the solve window T (design audits E31-3): no new task from T - 12 min; drain_at = T - drain_min
# (phase 1 aborted, its ids kept -> forced-close phase 2); every request aborted at T - 1 min (ORDER2's hard stop: the partial
# answer is parsed); the last minute is for writing. drain_min (E31_CFG) covers N x answer tokens at ~18 tok/s per stream.
ADMIT_STOP_BEFORE_END_SEC = 12 * 60
HARD_BEFORE_END_SEC = 60
PER_STREAM_TOK_S = 18.0                          # E28 K100 measured: 419.5 tok/s at 23.4 running (~18 tok/s per stream)
HEALTH_EVERY_SEC = 5.0
TOP_P, TOP_K, MIN_P = 0.95, 20, 0.0              # ORDER2 SAMPLING (+ min_p explicit: vLLM default 0.0)
STOP_THINKING_TEXT = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
                      "directly now.\n</think>\n\n")                          # ORDER2 nb[2] line 37, verbatim
# the Flash-Next tokenizer (klee100 tokenizer.json sha 06b95093, template c3cf9e34 = the 27B's): the spec's measured ids
FN_STOP_IDS = [271, 79939, 279, 6973, 854, 539, 279, 1156, 11, 353, 599, 310, 2873, 279, 6093, 3018, 383, 279, 7047, 5774,
               1381, 13, 198, 248069, 271]
FN_THINK_CLOSE_ID = 248069
FN_EOS_IDS = (248046, 248044)
CFG_DEFAULT = {"think": 104000, "answer": 8192, "window_min": 170, "workers": 24, "temperature": 0.6, "drain_min": 23}


def drain_need_min(max_tests, answer, tok_s=PER_STREAM_TOK_S):
    """The drain reserve that covers the largest live answer: max_tests x answer tokens at tok_s per stream (minutes)."""
    return int(max_tests) * int(answer) / float(tok_s) / 60.0

# user 2026-11-01: the 30 tasks (EVAL set, task_list) in three groups
GROUPS = {"main": ["2d0172a1", "35ab12c3", "409aa875", "446ef5d2", "4c3d4a41", "4c416de3", "5545f144", "7491f3cf", "7b5033c1",
                   "800d221b", "89565ca0", "9385bd28", "a47bf94d", "a6f40cea", "b6f77b65", "cbebaa4b"],
          "partial": ["142ca369", "247ef758", "36a08778", "581f7754", "71e489b6", "88e364bc", "a32d8b75", "f931b4a8"],
          "control": ["135a2760", "136b0064", "1818057f", "0934a4d8", "13e47133", "16b78196"]}
TASKS30 = [t for g in ("main", "partial", "control") for t in GROUPS[g]]
GROUP_OF = {t: g for g, ts in GROUPS.items() for t in ts}
# Exp 28 Experiment Results/EVAL120_K100_W4_Direct_Duck_per_task_2026-11-01.md, the 30 rows (1 = solved, 0.5 = partial, 0)
REF_COLUMNS = ["K100", "W4", "Direct27B", "Duck"]
REF_TABLE = {"2d0172a1": [0.0, 0.5, 0.0, 0.0], "35ab12c3": [0.0, 1.0, 0.0, 0.0], "409aa875": [0.0, 1.0, 1.0, 0.0],
             "446ef5d2": [0.0, 0.0, 0.5, 0.0], "4c3d4a41": [0.0, 1.0, 0.0, 0.0], "4c416de3": [0.0, 1.0, 0.0, 0.0],
             "5545f144": [0.0, 1.0, 0.0, 0.0], "7491f3cf": [0.0, 1.0, 1.0, 0.0], "7b5033c1": [0.0, 0.0, 1.0, 0.0],
             "800d221b": [0.0, 0.0, 0.0, 1.0], "89565ca0": [0.0, 1.0, 0.0, 1.0], "9385bd28": [0.0, 1.0, 0.5, 0.0],
             "a47bf94d": [0.0, 0.0, 1.0, 0.0], "a6f40cea": [0.0, 0.0, 1.0, 0.0], "b6f77b65": [0.0, 0.0, 0.5, 0.5],
             "cbebaa4b": [0.0, 0.0, 0.5, 0.0], "142ca369": [0.5, 1.0, 0.0, 1.0], "247ef758": [0.5, 1.0, 0.5, 0.5],
             "36a08778": [0.5, 1.0, 0.0, 1.0], "581f7754": [0.5, 0.5, 0.5, 0.5], "71e489b6": [0.5, 1.0, 0.5, 1.0],
             "88e364bc": [0.5, 0.5, 0.5, 0.0], "a32d8b75": [0.5, 1.0, 0.0, 0.0], "f931b4a8": [0.5, 0.0, 0.5, 1.0],
             "135a2760": [1.0, 1.0, 1.0, 1.0], "136b0064": [1.0, 1.0, 1.0, 1.0], "1818057f": [1.0, 1.0, 1.0, 1.0],
             "0934a4d8": [1.0, 0.0, 0.0, 0.0], "13e47133": [1.0, 1.0, 0.0, 0.0], "16b78196": [1.0, 1.0, 0.0, 0.0]}
E28_REFERENCE = {"source": "E28 K100 EVAL120 6 h (esfersami50, Duck): nb5 / e28 reports", "solver_tok_s": 419.5,
                 "server_running_mean": 23.41, "prompt_tokens_m": 101.4, "prefix_hit_tokens_m": 80.9, "fresh_prefill_tokens_m": 20.5,
                 "generated_tokens": 9062803}


# ==================================================================== prompt (ORDER2 nb[4] CELL 3, verbatim)
def render_grid(g):
    return "\n".join(" ".join(str(int(v)) for v in row) for row in g)


def build_user_message(train, test_input):
    parts = ["You are solving an abstract reasoning puzzle. Each example has an "
             "INPUT grid and its OUTPUT grid (digits 0-9 are colors; 0 is usually "
             "background). Infer the ONE rule that maps every input to its output, "
             "then apply it to the FINAL INPUT.\n"]
    for i, p in enumerate(train):
        parts.append("Example %d INPUT:\n%s\nExample %d OUTPUT:\n%s\n" %
                     (i + 1, render_grid(p["input"]), i + 1, render_grid(p["output"])))
    parts.append("FINAL INPUT:\n%s\n" % render_grid(test_input))
    parts.append("Reason step by step however you like. Then give ONLY the final "
                 "output grid as rows of space-separated digits inside a fenced "
                 "block:\n```grid\n<rows>\n```\nOutput nothing after the block.")
    return "\n".join(parts)


def build_user_message_multi(train, test_inputs):
    """E31 (user 2026-11-01): ONE prompt for a multi-test task. The minimal change of ORDER2's message: the final inputs are
    listed as 'FINAL INPUT k:' blocks and the closing asks for one labeled fenced block per final input, in order; every other
    sentence is ORDER2's."""
    parts = ["You are solving an abstract reasoning puzzle. Each example has an "
             "INPUT grid and its OUTPUT grid (digits 0-9 are colors; 0 is usually "
             "background). Infer the ONE rule that maps every input to its output, "
             "then apply it to the FINAL INPUT.\n"]
    for i, p in enumerate(train):
        parts.append("Example %d INPUT:\n%s\nExample %d OUTPUT:\n%s\n" %
                     (i + 1, render_grid(p["input"]), i + 1, render_grid(p["output"])))
    for k, g in enumerate(test_inputs):
        parts.append("FINAL INPUT %d:\n%s\n" % (k + 1, render_grid(g)))
    parts.append("Reason step by step however you like. Then give ONLY the final "
                 "output grids, one per FINAL INPUT in order, each as rows of space-separated digits inside its own "
                 "labeled fenced block:\n" + "\n".join("```grid %d\n<rows>\n```" % (k + 1) for k in range(len(test_inputs)))
                 + "\nOutput nothing after the last block.")
    return "\n".join(parts)


def task_message(task):
    """The user message of one task: ORDER2's verbatim for one test input, the multi-test variant otherwise."""
    tests = [q["input"] for q in task["test"]]
    return build_user_message(task["train"], tests[0]) if len(tests) == 1 else build_user_message_multi(task["train"], tests)


def order_score(task):                            # ORDER2 nb[4]
    return int(len(task["train"]) >= 3) + int(len(task["test"]) == 1)


def order2(tasks):
    """ORDER2 over tasks: higher score first, ties in sorted-task-id order (a task = one unit, E31)."""
    ids = sorted(tasks)
    return sorted(ids, key=lambda tid: (-order_score(tasks[tid]), ids.index(tid)))


def phase1_cap(prompt_len, think, answer, max_model_len=MAX_MODEL_LEN):
    """ORDER2 phase1_cap: min(think, max_model_len - prompt - answer - 256), >= 0."""
    return max(0, min(int(think), int(max_model_len) - int(prompt_len) - int(answer) - CONTEXT_SAFETY))


def task_seed(index, phase):                      # ORDER2: SEED + 10 * index + phase (index: sorted task ids)
    return SEED + 10 * int(index) + int(phase)


# ==================================================================== parsing (ORDER2 nb[5] CELL 4, verbatim)
def valid_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) and 1 <= len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))


_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
_SEP_ROW = re.compile(r"\[?\s*\d(?:\s*[, ]\s*\d)*\s*\]?,?")      # "1 0 2", "1,0,2", "[1, 0, 2]," (one digit per cell)
_JSON_ROW = re.compile(r"\[\s*\d(?:\s*,\s*\d)*\s*\]")


def _parse_row(s):
    s = s.strip()
    if not s:
        return None
    if re.fullmatch(r"\d+", s):                                  # contiguous digits, one per cell ("102")
        return [int(c) for c in s]
    if _SEP_ROW.fullmatch(s):
        return [int(c) for c in re.findall(r"\d", s)]
    return None


def _first_row_run(lines):
    rows = []
    for line in lines:
        r = _parse_row(line)
        if r is None:
            if rows:
                break
            continue
        rows.append(r)
    return rows if valid_grid(rows) else None


def _grid_from_obj(obj):
    if isinstance(obj, dict):
        for k in ("output", "grid", "test_output", "answer", "test_outputs"):
            if k in obj:
                return _grid_from_obj(obj[k])
        return None
    if valid_grid(obj):
        return obj
    if isinstance(obj, list) and obj and valid_grid(obj[-1]):     # a list of grids: the last one
        return obj[-1]
    return None


def _json_rows_grid(b):
    """v1.4: JSON with broken brackets (run 3: 6/12 blocks lacked one '['): rows are read one by one; a row followed by
    ']' ends a grid; the last grid wins"""
    grids, cur = [], []
    for m in _JSON_ROW.finditer(b):
        cur.append(json.loads(m.group(0)))
        if b[m.end():].lstrip().startswith("]"):
            grids.append(cur); cur = []
    if cur:
        grids.append(cur)
    for g in reversed(grids):
        if valid_grid(g):
            return g
    return None


def _grid_from_block(body):
    b = body.strip()
    if b[:1] in ("[", "{"):
        try:
            g = _grid_from_obj(json.loads(b))
        except Exception:
            g = None
        if g is None:
            g = _json_rows_grid(b)
        if g is not None:
            return g
    return _first_row_run(b.splitlines())


def parse_grid(text):
    """answer text after </think> -> grid or None: the last fenced block (any label) that holds a valid grid; else the
    last run of digit rows in the text (unfenced answer or a fence that was never closed)"""
    for _lab, body in reversed(_FENCE_RE.findall(text or "")):
        g = _grid_from_block(body)
        if g is not None:
            return g
    runs, cur = [], []
    for line in (text or "").splitlines():
        r = _parse_row(line)
        if r is None:
            if cur:
                runs.append(cur); cur = []
        else:
            cur.append(r)
    if cur:
        runs.append(cur)
    for run in reversed(runs):
        if valid_grid(run):
            return run
    return None


_LABEL_RE = re.compile(r"\s*grid\s*(\d+)\s*", re.I)


def parse_grids(text, n):
    """E31 (user 2026-11-01; design audits E31-1) -> (grids, mapping_source, fallback_reason), one entry per test input.
    n == 1: ORDER2's parse_grid, unchanged (source 'parse_grid'). n > 1: if ANY fenced block carries a recognized label
    'grid K', the mapping is by label ONLY (K in 1..n, a valid grid by ORDER2's _grid_from_block, the last such occurrence
    wins; never positional); with NO label, positional only when exactly n fenced blocks hold a valid grid. A test left
    without a grid -> source 'fallback' (make_attempts' test input / mirror for that test only) and its reason. Never a guess."""
    n = int(n)
    empty = not (text or "").strip()
    if n == 1:
        g = parse_grid(text)
        return [g], ["parse_grid" if g is not None else "fallback"], [None if g is not None else
                                                                       ("no_answer_text" if empty else "no_grid_parsed")]
    blocks = _FENCE_RE.findall(text or "")
    out, src, why = [None] * n, ["fallback"] * n, [None] * n
    labeled = [(int(m.group(1)), body) for lab, body in blocks for m in [_LABEL_RE.fullmatch(lab)] if m]
    if labeled:
        seen = {k for k, _ in labeled}
        for k in range(1, n + 1):
            for kk, body in reversed(labeled):
                if kk == k:
                    g = _grid_from_block(body)
                    if g is not None:
                        out[k - 1], src[k - 1] = g, "label"
                        break
            if out[k - 1] is None:
                why[k - 1] = "label_block_invalid" if k in seen else "label_missing"
    else:
        valid = [g for g in (_grid_from_block(b) for _l, b in blocks) if g is not None]
        if len(valid) == n:
            out, src = list(valid), ["positional"] * n
        else:
            why = [("no_answer_text" if empty else f"no_labels_{len(valid)}_valid_blocks_not_{n}")] * n
    return out, src, why


def make_attempts(test_input, grid):
    """ORDER2 make_attempts (per test input): attempt_1 = the model's grid; attempt_2 = the test input, or its mirror when
    the model returned the input"""
    inp = test_input
    cands = ([("model", grid)] if valid_grid(grid) else []) + [("test_input", inp), ("mirror", [list(reversed(r)) for r in inp])]
    uniq = []
    for src, g in cands:
        if all(g != u[1] for u in uniq):
            uniq.append((src, g))
    while len(uniq) < 2:                     # a palindromic input: mirror == input
        uniq.append(uniq[-1])
    return {"attempt_1": uniq[0][1], "attempt_2": uniq[1][1], "src_1": uniq[0][0], "src_2": uniq[1][0]}


def answer_text(ids, think_close_id, decode):
    """ORDER2 answer_text: the text after the LAST </think>, decoded with skip_special_tokens=True -> (text, k, closed)."""
    ids = list(ids)
    if think_close_id not in ids:
        return "", 0, False
    k = len(ids) - 1 - ids[::-1].index(think_close_id)
    return decode(ids[k + 1:], True), k, True


# ==================================================================== E18's streaming shim (e18_engine_cell.py E18Engine, adapted)
class _Completion:
    __slots__ = ("token_ids", "finish_reason")

    def __init__(self, ids, reason):
        self.token_ids, self.finish_reason = ids, reason


class _Output:
    __slots__ = ("request_id", "outputs", "finished")

    def __init__(self, rid, ids, reason):
        self.request_id, self.outputs, self.finished = rid, [_Completion(ids, reason)], reason is not None


def request_body(model_id, prompt_ids, max_tokens, temperature, seed):
    """Every sampling field explicit (the server fills a missing one from --override-generation-config: T 1.0, top_k 20, ...);
    E18's fields (prompt ids, stream, return_token_ids, skip_special_tokens False, top_k, seed) + min_p / penalties / ignore_eos
    at ORDER2's values (vLLM SamplingParams defaults)."""
    return {"model": model_id, "prompt": [int(x) for x in prompt_ids], "max_tokens": int(max_tokens),
            "temperature": float(temperature), "top_p": TOP_P, "top_k": TOP_K, "min_p": MIN_P,
            "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0, "seed": int(seed), "n": 1,
            "ignore_eos": False, "stream": True, "return_token_ids": True, "skip_special_tokens": False}


class StreamEngine:
    """E18Engine (Exp 18 e18_engine_cell.py, ran on stock vLLM 0.32): add_request starts one streaming /v1/completions per
    request in a thread; step() returns the token deltas that arrived (merged per request, order kept); abort_request
    closes the streams (vLLM aborts a request whose client disconnects) and drops their undelivered deltas. E31: the
    server relaunch is the notebook's E15B watchdog (not here); per-request timing is kept (first token, end)."""

    def __init__(self, base_url, model_id, req_log=None, idle_sec=None):
        self.base_url, self.model_id, self.idle_sec = base_url.rstrip("/"), model_id, idle_sec
        self.q, self.aborted, self.streams = _queue.Queue(), set(), {}
        self.lock = threading.Lock()
        self.req_log = Path(req_log) if req_log else None
        self.stats = {"added": 0, "finished": 0, "by_reason": {}, "errors": [], "completion_tokens": 0, "aborts": 0,
                      "missing_token_ids": 0}
        self.req_stats = {}

    def add_request(self, rid, prompt_ids, max_tokens, temperature, seed):
        body = request_body(self.model_id, prompt_ids, max_tokens, temperature, seed)
        st = {"rid": rid, "resp": None, "halt": False}
        with self.lock:
            self.aborted.discard(rid)
            self.streams[rid] = st
            self.stats["added"] += 1
            self.req_stats[rid] = {"t0": time.time(), "t_first": None, "t_end": None, "n": 0, "finish": None,
                                   "prompt": len(body["prompt"]), "max_tokens": body["max_tokens"], "seed": body["seed"]}
        threading.Thread(target=self._stream, args=(st, json.dumps(body).encode(), time.perf_counter()), daemon=True,
                         name=f"e31_stream_{rid}").start()

    def _stream(self, st, body, t0):
        rid, reason, err, n = st["rid"], None, None, 0
        try:
            req = _urlreq.Request(self.base_url + "/v1/completions", data=body, method="POST",
                                  headers={"Content-Type": "application/json"})
            st["resp"] = _urlreq.urlopen(req, timeout=self.idle_sec)
            if st["halt"]:                       # E18 rev 2: aborted while the request was being opened
                raise ConnectionAbortedError("aborted before the first chunk")
            for raw in st["resp"]:
                if st["halt"]:
                    break
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                d = json.loads(data)
                if "error" in d or d.get("object") == "error":
                    err = ("stream error: " + json.dumps(d.get("error") or d.get("message") or d))[:300]
                    break
                for c in d.get("choices") or []:
                    ids = c.get("token_ids")
                    if ids is None:
                        if c.get("text"):
                            with self.lock:
                                self.stats["missing_token_ids"] += 1
                        ids = []
                    ids = [int(x) for x in ids]
                    fr = c.get("finish_reason")
                    if fr is not None and reason is None:
                        reason = str(fr)
                    if ids:
                        with self.lock:
                            rs = self.req_stats.get(rid)
                            if rs is not None and rs["t_first"] is None:
                                rs["t_first"] = time.time()
                    n += len(ids)
                    if ids or fr is not None:
                        self.q.put((rid, ids, None if fr is None else str(fr)))
                if reason is not None:
                    break
        except Exception as e:
            if not st["halt"]:
                err = f"{type(e).__name__}: {e}"[:300]
        finally:                                 # E18 rev 2: close on every exit (vLLM aborts on disconnect)
            try:
                if st["resp"] is not None:
                    st["resp"].close()
            except Exception:
                pass
        if reason is None and not st["halt"]:
            reason = "error"
            self.q.put((rid, [], "error"))
        with self.lock:
            s = self.stats
            s["finished"] += 1
            rr = "abort" if st["halt"] else reason
            s["by_reason"][rr] = s["by_reason"].get(rr, 0) + 1
            s["completion_tokens"] += n
            if err and len(s["errors"]) < 50:
                s["errors"].append(f"{rid}: {err}")
            self.streams.pop(rid, None)
            rs = self.req_stats.get(rid)
            if rs is not None:
                rs.update(t_end=time.time(), n=n, finish=rr, error=err)
            if self.req_log is not None:
                try:
                    with open(self.req_log, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"rid": rid, "finish": rr, "tokens": n, "sec": round(time.perf_counter() - t0, 1),
                                            "error": err, **{k: (rs or {}).get(k) for k in ("t0", "t_first", "t_end", "prompt",
                                                                                         "max_tokens", "seed")}}) + "\n")
                except Exception:
                    pass

    def step(self, timeout=0.05):
        items = []
        try:
            items.append(self.q.get(timeout=timeout))
        except _queue.Empty:
            return []
        while True:
            try:
                items.append(self.q.get_nowait())
            except _queue.Empty:
                break
        merged = {}
        with self.lock:
            aborted = set(self.aborted)
        for rid, ids, reason in items:
            if rid in aborted:
                continue
            m = merged.setdefault(rid, [[], None])
            if m[1] is None:
                m[0].extend(ids)
                m[1] = reason
        return [_Output(rid, ids, reason) for rid, (ids, reason) in merged.items()]

    def abort_request(self, ids):
        for rid in list(ids):
            with self.lock:
                self.aborted.add(rid)
                st = self.streams.get(rid)
                self.stats["aborts"] += 1
            if st is None:
                continue
            st["halt"] = True
            try:   # shut the socket first so a reader blocked in recv returns, then close (vLLM aborts on disconnect)
                st["resp"].fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                if st["resp"] is not None:
                    st["resp"].close()
            except Exception:
                pass

    def live(self):
        with self.lock:
            return len(self.streams)


# ==================================================================== the solver (ORDER2 nb[7] CELL 6, one unit per task)
def _q(v, f):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(f * len(v)))] if v else None


def _health(base_url, timeout=5.0):
    try:
        _urlreq.urlopen(base_url.rstrip("/") + "/health", timeout=timeout).read()
        return True
    except Exception:
        return False


def solve_all_direct(tasks, order=None, *, out_dir, deadline, base_url, model_id, on_branch_done=None, workers=24,
                     capacity_probe=None, cfg=None, tok=None, solutions=None, report_dir=None, server_down=None,
                     metrics=None, sched_last=None, sampler_path=None, t0_pc=None, harness=False, ignored=None,
                     max_model_len=MAX_MODEL_LEN, health=None, max_error_retries=2, log=print):
    """E12A.solve_all_r6's contract (same keyword names, return {task: [record]}, on_branch_done(record) per finished task).
    tok = {"render": msgs -> text, "encode": text -> ids, "decode": (ids, skip_special) -> text, "eos_ids": [...]}.
    deadline = time.monotonic() of the solve end T. Phases inside T: no new task from T - 12 min, drain at T - drain_min
    (phase 1 aborted, ids kept -> forced-close phase 2), every request aborted at T - 1 min (partial answers parsed).
    The request continuation works on token ids only (never decode + re-tokenize). A request error is never a finish: the
    unit waits while the server is down / unhealthy (the notebook's E15B watchdog owns the relaunch), then continues
    (phase 1 error -> ORDER2's non-natural end = forced close; phase 2 error -> phase 2 again), at most max_error_retries;
    an unrecovered error ends the task as end_reason 'error'."""
    cfg = dict(CFG_DEFAULT, **(cfg or {}))
    think, answer, temp = int(cfg["think"]), int(cfg["answer"]), float(cfg["temperature"])
    drain_sec = float(cfg["drain_min"]) * 60.0
    if not harness:
        assert think >= THINK_FLOOR, ("user floor: thinking >= 60k (approval needed below)", think)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(report_dir) if report_dir else out_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    rec_path = report_dir / "e31_direct_records.jsonl"
    t_start_mono, t_start_pc, t_start_ep = time.monotonic(), time.perf_counter(), time.time()
    T0PC = t_start_pc if t0_pc is None else float(t0_pc)
    T_END = float(deadline)
    HARD = T_END - HARD_BEFORE_END_SEC
    DRAIN = T_END - drain_sec
    ADMIT = min(T_END - ADMIT_STOP_BEFORE_END_SEC, DRAIN)
    workers = max(1, int(workers))
    health = health or (lambda: _health(base_url))
    encode, decode, render = tok["encode"], tok["decode"], tok["render"]
    THINK_CLOSE_IDS = encode("</think>")
    assert len(THINK_CLOSE_IDS) == 1, THINK_CLOSE_IDS             # ORDER2 asserts
    THINK_CLOSE_ID = THINK_CLOSE_IDS[0]
    STOP_IDS = list(encode(STOP_THINKING_TEXT))                   # tokenized once (ORDER2 nb[3]); continuation = ids only
    assert THINK_CLOSE_ID in STOP_IDS, STOP_IDS
    if STOP_IDS != FN_STOP_IDS or THINK_CLOSE_ID != FN_THINK_CLOSE_ID:   # fail closed BEFORE any request (merge list item 4)
        why = {"status": "FAIL_CLOSED_TOKEN_IDS", "stop_ids": STOP_IDS, "spec_stop_ids": FN_STOP_IDS, "think_close_id": THINK_CLOSE_ID,
               "spec_think_close_id": FN_THINK_CLOSE_ID, "lib": LIB_VERSION}
        try:
            _rd = Path(report_dir) if report_dir else out_dir
            _rd.mkdir(parents=True, exist_ok=True)
            (_rd / "e31_direct_report.json").write_text(json.dumps(why, indent=1), encoding="utf-8")
        except Exception:
            pass
        raise RuntimeError("E31 FAIL_CLOSED_TOKEN_IDS: the tokenizer's stop ids / </think> id != the spec (no request sent): "
                           + json.dumps(why)[:600])
    EOS_IDS = set(int(x) for x in (tok.get("eos_ids") or []))
    assert EOS_IDS, "eos ids from the checkpoint's generation_config.json"
    task_ids = sorted(tasks)
    TASK_INDEX = {t: i for i, t in enumerate(task_ids)}            # seeds do not depend on timing (ORDER2 F13)
    ORDERED = order2(tasks)
    RANK = {t: i for i, t in enumerate(ORDERED)}
    max_tests = max([len(tasks[t]["test"]) for t in task_ids] or [1])
    SCHED = {"started_sec": round(t_start_pc - T0PC, 1), "admitted": 0, "phase2_continue": 0, "phase2_forced": 0, "drained": 0,
             "aborted_at_hard_stop": 0, "not_admitted": 0, "max_active": 0, "steps": 0, "workers": workers,
             "deadlines_rel_sec": {"admit_stop": round(ADMIT - t_start_mono, 1), "drain": round(DRAIN - t_start_mono, 1),
                                   "hard": round(HARD - t_start_mono, 1), "end": round(T_END - t_start_mono, 1)},
             "drain_min": cfg["drain_min"], "drain_need_min_max_task": round(drain_need_min(max_tests, answer), 2),
             "max_tests": max_tests, "request_errors": 0, "error_retries": 0, "error_finishes": 0, "server_down_polls": 0,
             "health_fail_polls": 0, "ignored_args": list(ignored or []), "lib": LIB_VERSION, "order": ORDERED, "cfg": cfg,
             "stop_ids": STOP_IDS, "stop_ids_equal_spec": STOP_IDS == FN_STOP_IDS, "think_close_id": THINK_CLOSE_ID,
             "eos_ids": sorted(EOS_IDS), "base_url": base_url, "errors": []}
    m0 = None
    try:
        m0 = metrics() if metrics else None
    except Exception as e:
        SCHED["errors"].append(f"metrics start: {type(e).__name__}: {e}"[:200])
    # ---- prompts (one per task)
    PROMPT_IDS, P1_CAPS, SKIPPED, NTEST, PROMPT_TEXT = {}, {}, {}, {}, {}
    for tid in task_ids:
        task = tasks[tid]
        NTEST[tid] = len(task["test"])
        text = render([{"role": "user", "content": task_message(task)}])
        ids = list(encode(text))
        cap = phase1_cap(len(ids), think, answer * NTEST[tid], max_model_len)   # room for answer x N test inputs
        if cap < min(MIN_THINK_BUDGET, think):     # ORDER2 v1.2: only prompts that leave < 4k thinking tokens are skipped
            SKIPPED[tid] = {"prompt_tokens": len(ids), "reason": "insufficient_context"}
            continue
        PROMPT_IDS[tid], P1_CAPS[tid], PROMPT_TEXT[tid] = ids, cap, text
    below_floor = {t: c for t, c in P1_CAPS.items() if c < THINK_FLOOR}
    SCHED.update(prompt_tokens={"median": _q([len(v) for v in PROMPT_IDS.values()], .5),
                                "max": max([len(v) for v in PROMPT_IDS.values()] or [0])},
                 below_floor_prompt_forbids=below_floor, skipped=SKIPPED, reduced_think=sum(c < think for c in P1_CAPS.values()))
    eng = StreamEngine(base_url, model_id, req_log=out_dir / "e31_requests.jsonl")
    STATE, RECORDS, OUT_RECORDS = {}, [], {t: [] for t in task_ids}
    gauges, gstate = [], {"last": -1e9, "h_t": -1e9, "h_ok": True}
    gen_total = [0]
    publish_seq = [0]

    def rid(tid, phase, attempt=0):
        return f"{tid}|{phase}|{attempt}"

    def admit(tid):
        st = {"tid": tid, "phase": 1, "prompt_ids": PROMPT_IDS[tid], "p1_ids": [], "p2_ids": [], "p2_prefix": [],
              "p1_finish": None, "p2_finish": None, "t_admit": time.perf_counter(), "t_admit_ep": time.time(), "t_done": None,
              "mode": None, "p2_cap": None, "pending": None, "errors": [], "retries": 0, "p2_rid": None, "p1_rid": rid(tid, 1)}
        STATE[tid] = st
        eng.add_request(st["p1_rid"], st["prompt_ids"], P1_CAPS[tid], temp, task_seed(TASK_INDEX[tid], 1))
        SCHED["admitted"] += 1

    def answer_cap(tid):                            # E31: answer x N test inputs (3 x 30x30 grids must fit), room-capped
        return answer * NTEST[tid]

    def start_phase2(st, resend=False):
        """ORDER2: think closed but answer cut -> continue; think never closed -> append the stop ids (token-exact). A resend
        (after a phase-2 request error) sends the same prefix again."""
        if not resend:
            closed = THINK_CLOSE_ID in st["p1_ids"]
            if not closed:                          # ORDER2 v1.1 (F8): an EOS before </think> must not precede the stop text
                while st["p1_ids"] and st["p1_ids"][-1] in EOS_IDS:
                    st["p1_ids"].pop()
            st["p2_prefix"] = [] if closed else list(STOP_IDS)
            st["mode"] = "continue" if closed else "forced_close"
        prefix = list(st["prompt_ids"]) + list(st["p1_ids"]) + list(st["p2_prefix"])
        room = max_model_len - len(prefix) - 8
        cap = min(int(answer_cap(st["tid"])), room)
        if cap < 64:
            return False
        st["phase"] = 2; st["p2_cap"] = cap; st["p2_ids"] = []; st["p2_prefix_len"] = len(prefix)
        if not resend:
            SCHED["phase2_continue" if st["mode"] == "continue" else "phase2_forced"] += 1
        st["p2_rid"] = rid(st["tid"], 2, st["retries"])
        eng.add_request(st["p2_rid"], prefix, cap, temp, task_seed(TASK_INDEX[st["tid"]], 2))
        return True

    def resend_phase1(st):
        """A phase-1 request error before the drain (merge list item 1): phase 1 is RESENT, token-exact (prompt ids + the kept
        phase-1 ids, trailing EOS stripped), cap = the phase-1 cap - the kept ids, the same seed. Under 64 tokens left = a
        length end -> the normal phase 2. Forced close only at / after the drain."""
        tid = st["tid"]
        while st["p1_ids"] and st["p1_ids"][-1] in EOS_IDS:
            st["p1_ids"].pop()
        cap = int(P1_CAPS[tid]) - len(st["p1_ids"])
        if cap < 64:
            st["p1_finish"] = "length"
            return start_phase2(st)
        st["p1_rid"] = rid(tid, 1, st["retries"])
        SCHED["phase1_resent"] = SCHED.get("phase1_resent", 0) + 1
        eng.add_request(st["p1_rid"], list(st["prompt_ids"]) + list(st["p1_ids"]), cap, temp, task_seed(TASK_INDEX[tid], 1))
        return True

    def finish(st, reason):
        tid = st["tid"]
        st["phase"] = "done"; st["t_done"] = time.perf_counter(); st["end_reason"] = reason
        all_ids = list(st["p1_ids"]) + list(st["p2_prefix"]) + list(st["p2_ids"])
        text, think_tokens, closed = answer_text(all_ids, THINK_CLOSE_ID, decode)
        grids, gsrc, gwhy = parse_grids(text, NTEST[tid])
        if not closed:
            gwhy = [w or None for w in gwhy]
            gwhy = ["no_think_close" if g is None else w for g, w in zip(grids, gwhy)]
        st.update({"answer_chars": len(text), "think_tokens": int(think_tokens if closed else len(st["p1_ids"])),
                   "think_closed": closed})
        try:      # user rule: the FULL thinking of every request is saved (never opened by the builder)
            d = out_dir / f"{tid}_b0"
            d.mkdir(parents=True, exist_ok=True)
            (d / "transcript.txt").write_text(
                "\n".join([f"### task={tid} end_reason={reason} mode={st['mode'] or 'natural'} p1_finish={st['p1_finish']} "
                           f"p2_finish={st['p2_finish']} prompt_tokens={len(st['prompt_ids'])} p1_tokens={len(st['p1_ids'])} "
                           f"p2_tokens={len(st['p2_ids'])} errors={st['errors']}", "### PROMPT", PROMPT_TEXT.get(tid, ""),
                           "### GENERATED (phase 1 + forced-close ids + phase 2, special tokens kept)",
                           decode(all_ids, False), "### ANSWER (after the last </think>)", text, ""]), encoding="utf-8")
            (d / "token_ids.json").write_text(json.dumps({"prompt": st["prompt_ids"], "p1": st["p1_ids"], "p2_prefix": st["p2_prefix"],
                                                          "p2": st["p2_ids"]}), encoding="utf-8")
        except Exception as e:
            SCHED["errors"].append(f"transcript {tid}: {type(e).__name__}: {e}"[:200])
        _record(st, grids, gsrc, gwhy)

    def _score(tid, qmap):
        if solutions is None or tid not in solutions:
            return None
        sol = solutions[tid]
        assert len(sol) == len(qmap) and [q["query_index"] for q in qmap] == list(range(len(sol))), (tid, len(sol), len(qmap))
        return [{"hit": int(q["attempt_1"] == sol[q["query_index"]] or q["attempt_2"] == sol[q["query_index"]]),
                 "pass1": int(q["attempt_1"] == sol[q["query_index"]])} for q in qmap]

    def _record(st, grids, gsrc, gwhy):
        tid = st["tid"]
        tests = [q["input"] for q in tasks[tid]["test"]]
        assert len(grids) == len(tests) == NTEST[tid], (tid, len(grids), len(tests))
        qmap = []                                   # query_index -> grid -> attempts (asserted before any gold scoring)
        for qi, (inp, g) in enumerate(zip(tests, grids)):
            a = make_attempts(inp, g)
            qmap.append({"query_index": qi, "grid": g, "mapping_source": gsrc[qi], "fallback_reason": gwhy[qi],
                         "attempt_1": a["attempt_1"], "attempt_2": a["attempt_2"], "src_1": a["src_1"], "src_2": a["src_2"]})
            assert (a["src_1"] == "model") == (g is not None) and (g is None or a["attempt_1"] == g), (tid, qi)
        sc = _score(tid, qmap)
        t1 = time.time()
        p1_rs = eng.req_stats.get(st["p1_rid"]) or {}
        p2_rs = eng.req_stats.get(st["p2_rid"]) or {} if st.get("p2_rid") else {}
        gen = len(st["p1_ids"]) + len(st["p2_ids"])
        e31 = {"mode": st["mode"] or "natural", "end_reason": st["end_reason"], "prompt_tokens": len(st["prompt_ids"]),
               "p1_tokens": len(st["p1_ids"]), "p2_tokens": len(st["p2_ids"]), "think_tokens": st["think_tokens"],
               "think_closed": st["think_closed"], "p1_finish": st["p1_finish"], "p2_finish": st["p2_finish"],
               "p1_cap": P1_CAPS.get(tid), "p2_cap": st["p2_cap"], "below_floor_prompt_forbids": tid in below_floor,
               "sec": round(st["t_done"] - st["t_admit"], 1), "answer_chars": st["answer_chars"], "n_test": NTEST[tid],
               "has_grid": [g is not None for g in grids], "mapping_source": gsrc, "fallback_reason": gwhy,
               "src": [[q["src_1"], q["src_2"]] for q in qmap], "score": sc, "request_errors": list(st["errors"]),
               "error_retries": st["retries"], "t_admit_min": round((st["t_admit"] - T0PC) / 60, 2),
               "t_done_min": round((st["t_done"] - T0PC) / 60, 2), "order_score": order_score(tasks[tid]), "admit_rank": RANK[tid],
               "group": GROUP_OF.get(tid), "seed_p1": task_seed(TASK_INDEX[tid], 1),
               "seed_p2": task_seed(TASK_INDEX[tid], 2) if st["p2_cap"] else None, "p2_prefix_len": st.get("p2_prefix_len"),
               "req": {"p1": {k: p1_rs.get(k) for k in ("t0", "t_first", "t_end", "n", "finish")},
                       "p2": {k: p2_rs.get(k) for k in ("t0", "t_first", "t_end", "n", "finish")} if p2_rs else None}}
        publish_seq[0] += 1
        rec = {"task_id": tid, "branch": 0, "status": "done", "publish_seq": publish_seq[0], "accepted": False,
               "generated_tokens": gen, "total_tokens": len(st["prompt_ids"]) + len(st["p2_prefix"]) + gen,
               "t0": st["t_admit_ep"], "t1": t1, "note": "", "nb5_verdict": None, "turns": 1 + int(bool(st["p2_cap"])),
               "failures": len(st["errors"]), "b_budget": int(P1_CAPS.get(tid) or 0) + answer_cap(tid), "view": "e31_direct",
               "usage": [{"prompt": len(st["prompt_ids"]), "cached": None, "completion": len(st["p1_ids"])}]
                        + ([{"prompt": int(st.get("p2_prefix_len") or 0), "cached": None, "completion": len(st["p2_ids"])}]
                           if st["p2_cap"] else []),
               "submissions": [{"test_outputs": list(grids), "accepted": False, "train_exact": [], "t": t1, "branch": 0,
                                "origin": "e31_direct", "name": "e31_direct"}],
               "e31_attempts": [{"attempt_1": q["attempt_1"], "attempt_2": q["attempt_2"]} for q in qmap],
               "e31_queries": [{k: q[k] for k in ("query_index", "mapping_source", "fallback_reason", "src_1", "src_2")} for q in qmap],
               "e31": e31}
        OUT_RECORDS[tid] = [rec]
        RECORDS.append(rec)
        try:
            (out_dir / f"{tid}_b0" / "branch_result.json").write_text(json.dumps(rec, default=str), encoding="utf-8")
        except Exception:
            pass
        try:
            with open(rec_path, "a", encoding="utf-8") as f:
                for q in qmap:
                    qi = q["query_index"]
                    row = dict(e31, unit=f"{tid}#{qi}", task_id=tid, query=qi, query_index=qi, has_grid=q["grid"] is not None,
                               mapping_source=q["mapping_source"], fallback_reason=q["fallback_reason"], src=[q["src_1"], q["src_2"]],
                               hit=(sc[qi]["hit"] if sc else None), pass1=(sc[qi]["pass1"] if sc else None),
                               finishes=[st["p1_finish"], st["p2_finish"]])
                    row.pop("score", None)
                    f.write(json.dumps(row) + "\n")
        except Exception as e:
            SCHED["errors"].append(f"records {tid}: {type(e).__name__}: {e}"[:200])
        log(f"[E31 {(time.perf_counter() - T0PC) / 60:6.1f} min] {tid} {e31['mode']:12s} {e31['end_reason']:14s} think "
            f"{e31['think_tokens']:6d} closed={e31['think_closed']} grids={sum(e31['has_grid'])}/{NTEST[tid]}" +
            (f" hits={sum(x['hit'] for x in sc)}/{len(sc)}" if sc else ""), flush=True)
        if on_branch_done is not None:
            try:
                on_branch_done(rec)
            except Exception as e:
                SCHED["errors"].append(f"on_branch_done {tid}: {type(e).__name__}: {e}"[:200])

    def gauge(now_mono, active, queue_left):
        if capacity_probe is None or now_mono - gstate["last"] < 30 or len(gauges) >= 3000:
            return
        gstate["last"] = now_mono
        try:
            snap = capacity_probe() or {}
        except Exception:
            snap = {}
        gauges.append([round(now_mono - t_start_mono), snap.get("kv_usage"), snap.get("mamba_usage"), snap.get("running"),
                       snap.get("queue"), len(active), queue_left])

    def server_ok(now_mono):
        """False while the notebook's watchdog relaunches the server (E15B SERVER_DOWN) or /health fails (polled <= 5 s)."""
        if server_down is not None and server_down.is_set():
            SCHED["server_down_polls"] += 1
            return False
        if now_mono - gstate["h_t"] >= HEALTH_EVERY_SEC:
            gstate["h_t"], gstate["h_ok"] = now_mono, bool(health())
            if not gstate["h_ok"]:
                SCHED["health_fail_polls"] += 1
        return gstate["h_ok"]

    def any_pending():
        return any(s["phase"] in (1, 2) and s["pending"] for s in STATE.values())

    queue_ = [t for t in ORDERED if t in PROMPT_IDS]   # ORDER2
    drained = False
    last_log = time.perf_counter()
    error = None
    try:
        while True:
            now = time.monotonic()
            active = [s for s in STATE.values() if s["phase"] in (1, 2)]
            if now >= HARD:                           # ORDER2 hard stop: abort everything, parse what exists
                live = [s["p1_rid"] if s["phase"] == 1 else s["p2_rid"] for s in active if not s["pending"]]
                if live:
                    eng.abort_request(live)
                for s in active:
                    SCHED["aborted_at_hard_stop"] += 1
                    finish(s, "hard_stop")
                break
            if now >= DRAIN and not drained:          # ORDER2 drain: phase 1 -> forced-close phase 2 (ids kept)
                drained = True
                p1 = [s for s in active if s["phase"] == 1]
                if [s for s in p1 if not s["pending"]]:
                    eng.abort_request([s["p1_rid"] for s in p1 if not s["pending"]])
                for s in p1:
                    if s["pending"]:
                        SCHED["drain_pending"] = SCHED.get("drain_pending", 0) + 1
                        continue                      # a unit waiting for the server keeps waiting (its phase 2 follows)
                    SCHED["drained"] += 1; s["p1_finish"] = "drained"
                    if not start_phase2(s):
                        finish(s, "drain_no_room")
                active = [s for s in STATE.values() if s["phase"] in (1, 2)]
            ok_srv = True
            if any_pending() or (queue_ and len(active) < workers and now < ADMIT and not drained):
                ok_srv = server_ok(now)
            if ok_srv:                                # units that waited for the server: continue (never a finish)
                for s in [s for s in active if s["pending"]]:
                    kind, s["pending"] = s["pending"], None
                    s["retries"] += 1; SCHED["error_retries"] += 1
                    if kind == "p1" and now < DRAIN and not drained:
                        ok_ = resend_phase1(s)            # before the drain: phase 1 continues (never a forced close)
                    else:
                        ok_ = start_phase2(s, resend=(kind == "p2"))   # p2 resend, or a p1 error at / after the drain
                    if not ok_:
                        finish(s, "no_room_for_answer")
                active = [s for s in STATE.values() if s["phase"] in (1, 2)]
            while ok_srv and queue_ and len(active) < workers and now < ADMIT and not drained:
                admit(queue_.pop(0)); active = [s for s in STATE.values() if s["phase"] in (1, 2)]
            SCHED["max_active"] = max(SCHED["max_active"], len(active))
            gauge(now, active, len(queue_))
            if not active:
                if not queue_ or now >= ADMIT or drained:
                    break                      # every task done (or no admission left): RETURN now, never wait for the window
                time.sleep(0.05)
                continue
            outs = eng.step(); SCHED["steps"] += 1
            for o in outs:
                tid, phase, _att = o.request_id.split("|"); phase = int(phase); st = STATE[tid]
                if st["phase"] != phase or o.request_id != (st["p1_rid"] if phase == 1 else st["p2_rid"]) or st["pending"]:
                    continue                          # stale outputs after an abort / resend are ignored (ORDER2)
                comp = o.outputs[0] if o.outputs else None
                if comp is not None:
                    new = list(comp.token_ids); gen_total[0] += len(new)
                    (st["p1_ids"] if phase == 1 else st["p2_ids"]).extend(new)
                if o.finished:
                    fr = comp.finish_reason if comp is not None else None
                    if fr == "error":                 # never a finish: wait for the server, then continue (bounded)
                        SCHED["request_errors"] += 1
                        st["errors"].append({"phase": phase, "rid": o.request_id, "t_sec": round(time.perf_counter() - T0PC, 1)})
                        if phase == 1:
                            st["p1_finish"] = "error"
                        else:
                            st["p2_finish"] = "error"
                        if st["retries"] >= int(max_error_retries):
                            SCHED["error_finishes"] += 1
                            finish(st, "error")
                        else:
                            st["pending"] = "p1" if phase == 1 else "p2"
                            gstate["h_t"] = -1e9      # re-check the server before the unit continues
                        continue
                    if phase == 1:
                        st["p1_finish"] = fr
                        closed = THINK_CLOSE_ID in st["p1_ids"]
                        if closed and fr != "length":
                            finish(st, "natural")
                        elif not start_phase2(st):
                            finish(st, "no_room_for_answer")
                    else:
                        st["p2_finish"] = fr; finish(st, st["mode"])
            if time.perf_counter() - last_log > 120.0:
                last_log = time.perf_counter(); el = last_log - t_start_pc
                log(f"  E31 .. {el / 60:6.1f} min | active {len(active)} | queued {len(queue_)} | done {len(RECORDS)} | "
                    f"gen {gen_total[0]} tok | {gen_total[0] / max(el, 1e-6):.0f} tok/s", flush=True)
    except BaseException as e:            # ORDER2 v1.1 (F2): keep every finished task, report the error
        import traceback
        error = traceback.format_exc()[-4000:]
        log("E31 SCHEDULER ERROR - keeping finished tasks:", error[-2000:], flush=True)
        try:
            _act = [s["p1_rid"] if s["phase"] == 1 else s["p2_rid"] for s in STATE.values() if s["phase"] in (1, 2)]
            if _act:
                eng.abort_request([x for x in _act if x])
        except BaseException:
            pass
        if not isinstance(e, Exception):
            raise
    wall = round(time.perf_counter() - t_start_pc, 1)
    SCHED.update(not_admitted=len(queue_), not_admitted_tasks=list(queue_), wall_sec=wall, gen_tokens=gen_total[0],
                 tok_per_sec=round(gen_total[0] / max(wall, 1e-6), 1), error=error, engine=dict(eng.stats),
                 ended_before_deadline_sec=round(T_END - time.monotonic(), 1), returned_early=time.monotonic() < HARD,
                 live_streams_at_return=eng.live())
    m1 = None
    try:
        m1 = metrics() if metrics else None
    except Exception as e:
        SCHED["errors"].append(f"metrics end: {type(e).__name__}: {e}"[:200])
    rep = build_report(RECORDS, tasks, solutions, SCHED, eng.req_stats, m0, m1, sampler_path, t_start_ep, time.time(), SKIPPED,
                       think=think)
    try:
        (report_dir / "e31_direct_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    except Exception as e:
        SCHED["errors"].append(f"report write: {type(e).__name__}: {e}"[:200])
    if sched_last is not None:              # E12A.SCHED_LAST: what e21_cell6 / CELL 7 read after the solver (Duck's keys)
        sched_last.clear()
        sched_last.update(e31=SCHED, gauges=gauges[-3000:], gauge_fields=["t", "kv", "mamba", "server_running", "server_queue",
                                                                          "workers_running", "heap"],
                          stragglers=0, worker_errors=int(bool(error)), setup_failed=len(SKIPPED), workers=workers,
                          tasks_initialized=SCHED["admitted"], tasks_with_request_sent=SCHED["admitted"], stage1_left=len(queue_),
                          heap_left=len(queue_), events=[], allowed=None, sessions={t: "done" for t in OUT_RECORDS if OUT_RECORDS[t]})
    log("E31 solver done:", json.dumps({k: SCHED.get(k) for k in ("admitted", "phase2_continue", "phase2_forced", "drained",
                                                                   "aborted_at_hard_stop", "not_admitted", "max_active", "wall_sec",
                                                                   "gen_tokens", "tok_per_sec", "request_errors", "error_retries",
                                                                   "returned_early")}), flush=True)
    return OUT_RECORDS


def build_submission_direct(tasks, records):
    """E31's DIRECT-only selector (design audits E31-2): per task the LATEST direct record's ORDER2 make_attempts, one row per
    test input; a task without a direct record (not admitted / not finished) = the identity (ORDER2 write_submission). No Duck
    candidate, no harvested program, no CPU rule ever enters."""
    out = {}
    for tid, task in tasks.items():
        rs = [r for r in (records.get(tid) or []) if isinstance(r, dict) and r.get("e31_attempts")]
        att = max(rs, key=lambda r: int(r.get("publish_seq") or 0))["e31_attempts"] if rs else None
        if att is not None and len(att) == len(task["test"]):
            out[tid] = [{"attempt_1": a["attempt_1"], "attempt_2": a["attempt_2"]} for a in att]
        else:
            out[tid] = [{"attempt_1": q["input"], "attempt_2": q["input"]} for q in task["test"]]
    return out


# ==================================================================== the E31 report
def _sampler_tok_s(path, ep0, ep1):
    try:
        rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    except Exception:
        return None
    rows = [r for r in rows if isinstance(r.get("gen"), (int, float)) and isinstance(r.get("t"), (int, float))
            and ep0 - 1 <= r["t"] <= ep1 + 1]
    if len(rows) < 2:
        return {"samples": len(rows), "tok_s": None}
    a, b = rows[0], rows[-1]
    dt = b["t"] - a["t"]
    run = [r.get("running") for r in rows if isinstance(r.get("running"), int)]
    return {"samples": len(rows), "tok_s": round((b["gen"] - a["gen"]) / dt, 1) if dt > 0 else None,
            "running_mean": round(sum(run) / len(run), 2) if run else None, "running_max": max(run) if run else None}


def build_report(records, tasks, solutions, sched, req_stats, m0, m1, sampler_path, ep0, ep1, skipped=None, think=None):
    by = {r["task_id"]: r for r in records}
    tids = sorted(tasks)
    groups = {}
    per_task = {}
    for t in tids:
        r = by.get(t)
        e = (r or {}).get("e31") or {}
        sc = e.get("score")
        n = len(tasks[t]["test"])
        hits = sum(x["hit"] for x in sc) if sc else (None if solutions is None else 0)
        pass1 = sum(x["pass1"] for x in sc) if sc else (None if solutions is None else 0)
        ref = REF_TABLE.get(t)
        e31_mark = None if hits is None else (1.0 if hits == n else (0.5 if hits else 0.0))
        per_task[t] = {"group": GROUP_OF.get(t), "queries": n, "multi_test": n > 1, "query_hits": hits, "pass1": pass1,
                       "fully_solved": (hits == n) if hits is not None else None, "e31": e31_mark,
                       "ref": dict(zip(REF_COLUMNS, ref)) if ref else None, "finished": r is not None,
                       "mode": e.get("mode"), "end_reason": e.get("end_reason"), "think_tokens": e.get("think_tokens"),
                       "p1_tokens": e.get("p1_tokens"), "think_closed": e.get("think_closed"),
                       "generated_tokens": (r or {}).get("generated_tokens"), "prompt_tokens": e.get("prompt_tokens"),
                       "has_grid": e.get("has_grid"), "mapping_source": e.get("mapping_source"),
                       "fallback_reason": e.get("fallback_reason"), "sec": e.get("sec"), "request_errors": len(e.get("request_errors") or []),
                       "skipped": (skipped or {}).get(t)}
        keys = [GROUP_OF.get(t) or "other", "all", "multi_test" if n > 1 else "single_test",
                f"{GROUP_OF.get(t) or 'other'}_{'multi' if n > 1 else 'single'}"]
        for g in keys:
            s = groups.setdefault(g, {"tasks": 0, "fully_solved_tasks": 0, "task_score_sum": 0.0, "query_hits": 0, "queries": 0,
                                      "pass1": 0, "finished": 0})
            s["tasks"] += 1
            s["queries"] += n
            s["finished"] += int(r is not None)
            if hits is not None:
                s["fully_solved_tasks"] += int(hits == n)
                s["task_score_sum"] += hits / max(1, n)
                s["query_hits"] += hits
                s["pass1"] += pass1
    for s in groups.values():
        s["task_norm_pct"] = round(100.0 * s.pop("task_score_sum") / s["tasks"], 2) if (s["tasks"] and solutions is not None) else None
    cmp_ = {}
    if solutions is not None:
        for j, col in enumerate(REF_COLUMNS):
            rows = [(per_task[t]["e31"], REF_TABLE[t][j]) for t in tids if t in REF_TABLE and per_task[t]["e31"] is not None]
            cmp_[col] = {"tasks": len(rows), "e31_fully_solved": sum(1 for a, _ in rows if a == 1.0),
                         "ref_fully_solved": sum(1 for _, b in rows if b == 1.0),
                         "ref_partial": sum(1 for _, b in rows if b == 0.5),
                         "both_fully_solved": sum(1 for a, b in rows if a == 1.0 and b == 1.0),
                         "e31_only": sorted(t for t in tids if t in REF_TABLE and per_task[t]["e31"] == 1.0 and REF_TABLE[t][j] != 1.0),
                         "ref_only": sorted(t for t in tids if t in REF_TABLE and per_task[t]["e31"] is not None
                                            and per_task[t]["e31"] != 1.0 and REF_TABLE[t][j] == 1.0)}
    es = [r.get("e31") or {} for r in records]
    modes, ends, msrc = {}, {}, {}
    for e in es:
        modes[e.get("mode")] = modes.get(e.get("mode"), 0) + 1
        ends[e.get("end_reason")] = ends.get(e.get("end_reason"), 0) + 1
        for m in e.get("mapping_source") or []:
            msrc[m] = msrc.get(m, 0) + 1
    # E31-4: thinking coverage (record-only): a task is covered at X when its thinking closed naturally or ran >= X tokens
    cov = {}
    for x in (60000, 78000, 104000):
        reached = [t for t in tids if (per_task[t]["p1_tokens"] or 0) >= x]
        covered = [t for t in tids if per_task[t]["finished"] and (per_task[t]["mode"] == "natural" or (per_task[t]["p1_tokens"] or 0) >= x
                                                                   or (per_task[t]["think_closed"] and per_task[t]["end_reason"] in ("continue", "natural")))]
        cov[str(x)] = {"p1_reached": len(reached), "covered_closed_or_reached": len(covered),
                       "cut_before": sorted(t for t in tids if per_task[t]["finished"] and t not in covered),
                       "not_finished": sorted(t for t in tids if not per_task[t]["finished"])}
    gen = [r.get("generated_tokens") or 0 for r in records]
    rs = [v for v in (req_stats or {}).values() if v.get("t_end")]
    per_req = [v["n"] / (v["t_end"] - v["t_first"]) for v in rs if v.get("t_first") and v.get("n", 0) > 1 and v["t_end"] > v["t_first"]]
    ttft = [v["t_first"] - v["t0"] for v in rs if v.get("t_first")]
    wall = (sched or {}).get("wall_sec") or 0
    srv = {}
    if isinstance(m0, dict) and isinstance(m1, dict):
        for k in ("vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:prefix_cache_hits_total",
                  "vllm:prefix_cache_queries_total", "vllm:num_preemptions_total"):
            if isinstance(m0.get(k), (int, float)) and isinstance(m1.get(k), (int, float)):
                srv[k] = m1[k] - m0[k]
        if "vllm:prompt_tokens_total" in srv and "vllm:prefix_cache_hits_total" in srv:
            srv["fresh_prefill_tokens"] = srv["vllm:prompt_tokens_total"] - srv["vllm:prefix_cache_hits_total"]
        if "vllm:generation_tokens_total" in srv and wall:
            srv["server_decode_tok_s"] = round(srv["vllm:generation_tokens_total"] / wall, 1)
    return {"lib": LIB_VERSION, "status": "SCORED" if solutions is not None else "UNSCORED_RERUN",
            "tasks_planned": len(tids), "tasks_finished": len(records), "queries_planned": sum(len(tasks[t]["test"]) for t in tids),
            "groups": groups, "per_task": per_task, "comparison_vs_reference": cmp_, "reference_columns": REF_COLUMNS,
            "reference_note": "1 = fully solved, 0.5 = partial (some test inputs), 0 = not; Exp 28 per-task table", "modes": modes,
            "end_reasons": ends, "mapping_sources": msrc, "think_coverage": cov,
            "think_closed_naturally": sum(1 for e in es if e.get("mode") == "natural"),
            "below_floor_prompt_forbids": (sched or {}).get("below_floor_prompt_forbids"),
            "mean_think_tokens": round(sum(e.get("think_tokens") or 0 for e in es) / len(es), 1) if es else None,
            "mean_generated_tokens": round(sum(gen) / len(gen), 1) if gen else None,
            "mean_prompt_tokens": round(sum(e.get("prompt_tokens") or 0 for e in es) / len(es), 1) if es else None,
            "generated_tokens_p50_p90": [_q(gen, .5), _q(gen, .9)],
            "speed": {"solve_wall_sec": wall, "client_generated_tokens": sum(gen),
                      "aggregate_decode_tok_s": round(sum(gen) / wall, 1) if wall else None,
                      "per_request_tok_s_p50": round(_q(per_req, .5), 2) if per_req else None,
                      "per_request_tok_s_mean": round(sum(per_req) / len(per_req), 2) if per_req else None,
                      "ttft_sec_p50": round(_q(ttft, .5), 2) if ttft else None, "requests": len(rs),
                      "server_counters_delta": srv, "sampler": _sampler_tok_s(sampler_path, ep0, ep1) if sampler_path else None,
                      "returned_early": (sched or {}).get("returned_early"),
                      "ended_before_deadline_sec": (sched or {}).get("ended_before_deadline_sec")},
            "e28_duck_reference": E28_REFERENCE, "sched": sched}
